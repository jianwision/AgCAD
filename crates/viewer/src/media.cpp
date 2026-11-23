#include "draw.hpp"
#include <chrono>
#include <cstdio>
#include <deque>
#include <gst/d3d11/gstd3d11.h>
#include <gst/gst.h>
#include <gst/sdp/sdp.h>
#include <gst/webrtc/webrtc.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct Frame {
  uint32_t timestamp, sequence;
  uint64_t scene;
  Uniforms matrix;
};
struct Peer {
  GstElement *images = nullptr;
  GstElement *pipeline = nullptr, *rtc = nullptr;
  GstWebRTCDataChannel *channel = nullptr;
  std::mutex lock;
  Renderer renderer;
  Camera camera;
  uint32_t skybox = 0;
  std::vector<Vertex> vertices, marker;
  uint64_t marker_id = 0;
  std::vector<std::array<float, 8>> colors;
  uint64_t palette_version = 0;
  uint64_t scene = 0;
  uint32_t selected = UINT32_MAX;
  std::deque<Frame> frames;
  std::string error;
  ~Peer() {
    if (pipeline)
      gst_element_set_state(pipeline, GST_STATE_NULL);
    if (images)
      gst_object_unref(images);
    if (channel) {
      g_signal_handlers_disconnect_by_data(channel, this);
      g_object_unref(channel);
    }
    if (rtc) {
      g_signal_handlers_disconnect_by_data(rtc, this);
      gst_object_unref(rtc);
    }
    if (pipeline)
      gst_object_unref(pipeline);
  }
};
static gboolean draw(GstElement *, GstD3D11Device *device,
                     ID3D11RenderTargetView *target, guint64 pts, guint64,
                     Peer *peer) {
  std::lock_guard<std::mutex> guard(peer->lock);
  if (!peer->scene || (!peer->vertices.empty() && peer->colors.empty()) || !peer->error.empty())
    return TRUE;
  auto u = transform(peer->camera, peer->selected);
  u.skybox = peer->skybox;
  try {
    peer->renderer.draw(gst_d3d11_device_get_device_handle(device),
                        gst_d3d11_device_get_device_context_handle(device),
                        target, peer->scene, peer->vertices, u, peer->colors,
                        peer->palette_version, peer->marker, peer->marker_id);
    // The payloader has a zero timestamp offset and uses running time (90 kHz).
    peer->frames.push_back(
        {static_cast<uint32_t>(gst_util_uint64_scale(pts, 90000, GST_SECOND)),
         peer->camera.sequence, peer->scene, u});
    if (peer->frames.size() > 240)
      peer->frames.pop_front();
  } catch (const std::exception &e) {
    peer->error = e.what();
  }
  return TRUE;
}
static void camera_message(GstWebRTCDataChannel *, gchar *text, Peer *peer) {
  if (!text)
    return;
  Camera c;
  if (sscanf_s(text, "%u %f %f %f %f %f", &c.sequence, &c.yaw, &c.pitch,
               &c.zoom, &c.pan_x, &c.pan_y) != 6)
    return;
  if (!std::isfinite(c.yaw) || !std::isfinite(c.pitch) ||
      !std::isfinite(c.zoom) || !std::isfinite(c.pan_x) ||
      !std::isfinite(c.pan_y))
    return;
  c.zoom = std::clamp(c.zoom, .05f, 100.f);
  std::lock_guard<std::mutex> guard(peer->lock);
  if (c.sequence > peer->camera.sequence)
    peer->camera = c;
}
static void channel_open(GstElement *, GstWebRTCDataChannel *channel,
                         Peer *peer) {
  std::lock_guard<std::mutex> guard(peer->lock);
  if (peer->channel)
    return;
  peer->channel = static_cast<GstWebRTCDataChannel *>(g_object_ref(channel));
  g_signal_connect(channel, "on-message-string", G_CALLBACK(camera_message),
                   peer);
}
static void done(GstPromise *p) {
  if (gst_promise_wait(p) != GST_PROMISE_RESULT_REPLIED) {
    gst_promise_unref(p);
    throw std::runtime_error("WebRTC negotiation interrupted");
  }
  auto reply = gst_promise_get_reply(p);
  if (reply && gst_structure_has_field(reply, "error")) {
    GError *err = nullptr;
    gst_structure_get(reply, "error", G_TYPE_ERROR, &err, nullptr);
    std::string message = err ? err->message : "SDP rejected";
    g_clear_error(&err);
    gst_promise_unref(p);
    throw std::runtime_error(message);
  }
}
static void description(Peer &peer, const char *signal,
                        GstWebRTCSessionDescription *sdp) {
  auto p = gst_promise_new();
  g_signal_emit_by_name(peer.rtc, signal, sdp, p);
  done(p);
  gst_promise_unref(p);
}
static int h264_payload(const GstSDPMessage *sdp) {
  for (guint i = 0; i < gst_sdp_message_medias_len(sdp); i++) {
    auto media = gst_sdp_message_get_media(sdp, i);
    if (strcmp(gst_sdp_media_get_media(media), "video") != 0)
      continue;
    for (guint n = 0; n < gst_sdp_media_formats_len(media); n++) {
      int pt = atoi(gst_sdp_media_get_format(media, n));
      if (pt < 96 || pt > 127)
        continue;
      auto caps = gst_sdp_media_get_caps_from_media(media, pt);
      if (!caps)
        continue;
      auto s = gst_caps_get_structure(caps, 0);
      auto codec = gst_structure_get_string(s, "encoding-name"),
           mode = gst_structure_get_string(s, "packetization-mode");
      auto profile = gst_structure_get_string(s, "profile-level-id");
      bool usable = codec && !g_ascii_strcasecmp(codec, "H264") && mode &&
                    !strcmp(mode, "1") &&
                    (!profile || g_str_has_prefix(profile, "42"));
      gst_caps_unref(caps);
      if (usable)
        return pt;
    }
  }
  throw std::runtime_error(
      "Browser must offer H.264 baseline with packetization-mode 1");
}
extern "C" Peer *view_open(const char *offer, char **answer, char **error) {
  try {
    gst_init(nullptr, nullptr);
    std::unique_ptr<Peer> peer(new Peer);
    GstSDPMessage *msg = nullptr;
    if (gst_sdp_message_new_from_text(offer, &msg) != GST_SDP_OK)
      throw std::runtime_error("Invalid SDP offer");
    using Desc =
        std::unique_ptr<GstWebRTCSessionDescription,
                        decltype(&gst_webrtc_session_description_free)>;
    Desc remote(
        gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, msg),
        gst_webrtc_session_description_free);
    auto pt = std::to_string(h264_payload(msg));
    std::string pipeline =
        "webrtcbin name=peer bundle-policy=max-bundle latency=0 "
        "d3d11testsrc is-live=true pattern=black ! "
        "video/"
        "x-raw(memory:D3D11Memory),format=BGRA,width=1920,height=1080,"
        "framerate=60/1 "
        "! d3d11overlay name=viewport ! queue max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 leaky=downstream "
        "! nvd3d11h264enc tune=ultra-low-latency preset=p2 bframes=0 "
        "zerolatency=true bitrate=12000 gop-size=60 "
        "! rtph264pay timestamp-offset=0 perfect-rtptime=false "
        "config-interval=-1 pt=" +
        pt +
        " ! "
        "application/"
        "x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=" +
        pt + " ! peer.";
    GError *parse = nullptr;
    peer->pipeline = gst_parse_launch(pipeline.c_str(), &parse);
    if (parse) {
      std::string text = parse->message;
      g_error_free(parse);
      throw std::runtime_error(text);
    }
    auto overlay = gst_bin_get_by_name(GST_BIN(peer->pipeline), "viewport");
    g_signal_connect(overlay, "draw", G_CALLBACK(draw), peer.get());
    gst_object_unref(overlay);
    peer->rtc = gst_bin_get_by_name(GST_BIN(peer->pipeline), "peer");
    if (const char *stun = g_getenv("AGCAD_STUN_SERVER"))
      g_object_set(peer->rtc, "stun-server", stun, nullptr);
    if (const char *turn = g_getenv("AGCAD_TURN_SERVER"))
      g_object_set(peer->rtc, "turn-server", turn, nullptr);
    g_signal_connect(peer->rtc, "on-data-channel", G_CALLBACK(channel_open),
                     peer.get());
    if (gst_element_set_state(peer->pipeline, GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE)
      throw std::runtime_error("Cannot start native video pipeline");
    description(*peer, "set-remote-description", remote.get());
    auto p = gst_promise_new();
    g_signal_emit_by_name(peer->rtc, "create-answer", nullptr, p);
    done(p);
    GstWebRTCSessionDescription *raw = nullptr;
    gst_structure_get(gst_promise_get_reply(p), "answer",
                      GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &raw, nullptr);
    gst_promise_unref(p);
    if (!raw)
      throw std::runtime_error("Missing SDP answer");
    Desc local(raw, gst_webrtc_session_description_free);
    description(*peer, "set-local-description", local.get());
    for (int i = 0; i < 250; i++) {
      GstWebRTCICEGatheringState state;
      g_object_get(peer->rtc, "ice-gathering-state", &state, nullptr);
      if (state == GST_WEBRTC_ICE_GATHERING_STATE_COMPLETE)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    raw = nullptr;
    g_object_get(peer->rtc, "local-description", &raw, nullptr);
    if (!raw)
      throw std::runtime_error("Missing local ICE description");
    Desc gathered(raw, gst_webrtc_session_description_free);
    *answer = gst_sdp_message_as_text(gathered->sdp);
    return peer.release();
  } catch (const std::exception &e) {
    *error = g_strdup(e.what());
    return nullptr;
  }
}
// The HTTP transport renders only while a still is requested. The JPEG PTS
// selects the same recorded camera matrix used by video-frame picking.
extern "C" Peer *view_open_images(char **error) {
  try {
    gst_init(nullptr, nullptr);
    auto p = std::make_unique<Peer>();
    GError *parse = nullptr;
    p->pipeline = gst_parse_launch(
        "d3d11testsrc is-live=true pattern=black ! "
        "video/"
        "x-raw(memory:D3D11Memory),format=BGRA,width=1920,height=1080,"
        "framerate=30/1 "
        "! d3d11overlay name=viewport ! d3d11download ! videoconvert ! "
        // JPEG decoders expect full-range BT.601 YCbCr (sRGB transfer, BT.709 primaries).
        "video/x-raw,format=I420,colorimetry=(string)1:4:7:1 ! jpegenc quality=80 "
        "! appsink name=images async=false sync=false max-buffers=1 drop=true",
        &parse);
    if (parse) {
      std::string message = parse->message;
      g_error_free(parse);
      throw std::runtime_error(message);
    }
    auto overlay = gst_bin_get_by_name(GST_BIN(p->pipeline), "viewport");
    g_signal_connect(overlay, "draw", G_CALLBACK(draw), p.get());
    gst_object_unref(overlay);
    p->images = gst_bin_get_by_name(GST_BIN(p->pipeline), "images");
    return p.release();
  } catch (const std::exception &e) {
    *error = g_strdup(e.what());
    return nullptr;
  }
}
extern "C" void *view_image(Peer *p, const float pose[5], Frame *frame,
                            size_t *length) {
  uint32_t sequence;
  {
    std::lock_guard<std::mutex> guard(p->lock);
    sequence = p->camera.sequence + 1;
    p->camera = {pose[0], pose[1], pose[2], pose[3], pose[4], sequence};
  }
  if (gst_element_set_state(p->pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE)
    return nullptr;
  void *bytes = nullptr;
  for (int i = 0; i < 4 && !bytes; i++) {
    GstSample *sample = nullptr;
    g_signal_emit_by_name(p->images, "try-pull-sample",
                          static_cast<guint64>(GST_SECOND), &sample);
    if (!sample)
      break;
    auto buf = gst_sample_get_buffer(sample);
    uint32_t pts = static_cast<uint32_t>(
        gst_util_uint64_scale(GST_BUFFER_PTS(buf), 90000, GST_SECOND));
    {
      std::lock_guard<std::mutex> guard(p->lock);
      for (auto f = p->frames.rbegin(); f != p->frames.rend(); ++f) {
        if (f->timestamp != pts || f->sequence != sequence)
          continue;
        *frame = *f;
        gst_buffer_extract_dup(buf, 0, gst_buffer_get_size(buf), &bytes,
                               length);
        break;
      }
    }
    gst_sample_unref(sample);
  }
  gst_element_set_state(p->pipeline, GST_STATE_PAUSED);
  return bytes;
}
extern "C" void view_scene(Peer *p, uint64_t scene, const Vertex *vertices,
                           size_t count) {
  std::lock_guard<std::mutex> guard(p->lock);
  if (p->scene != scene) {
    p->vertices.assign(vertices, vertices + count);
    p->scene = scene;
    p->selected = UINT32_MAX;
    p->marker.clear();
    p->marker_id = 0;
  }
}
extern "C" void view_select(Peer *p, uint32_t face, uint64_t id, const Vertex *marker, size_t count) {
  std::lock_guard<std::mutex> guard(p->lock);
  p->selected = face;
  if (id != p->marker_id) {
    p->marker.assign(marker, marker + count);
    p->marker_id = id;
  }
}
extern "C" void view_palette(Peer *p, const std::array<float, 8> *colors,
                             size_t count) {
  std::lock_guard<std::mutex> guard(p->lock);
  p->colors.assign(colors, colors + count);
  ++p->palette_version;
}
extern "C" void view_skybox(Peer *p, uint32_t preset) {
  std::lock_guard<std::mutex> guard(p->lock);
  p->skybox = preset;
}
extern "C" bool view_frame(Peer *p, uint32_t timestamp, Frame *out) {
  std::lock_guard<std::mutex> guard(p->lock);
  for (auto it = p->frames.rbegin(); it != p->frames.rend(); ++it)
    if (it->timestamp == timestamp) {
      *out = *it;
      return true;
    }
  return false;
}
extern "C" char *view_error(Peer *p) {
  std::lock_guard<std::mutex> guard(p->lock);
  auto bus = gst_element_get_bus(p->pipeline);
  auto msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
  if (msg) {
    GError *e = nullptr;
    gst_message_parse_error(msg, &e, nullptr);
    p->error = e->message;
    g_error_free(e);
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
  return p->error.empty() ? nullptr : g_strdup(p->error.c_str());
}
extern "C" void view_free_string(char *s) { g_free(s); }
extern "C" void view_close(Peer *p) { delete p; }
