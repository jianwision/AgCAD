//! Fuzzer-found layered-solid regressions (each was a distinct bug class); every program must
//! build a valid B-rep whose tessellation is watertight with the exact volume.

use agcad_conformance::{check_program, FuzzOp::*, FuzzProgram};

fn ok(p: FuzzProgram) {
    if let Err(e) = check_program(&p, false) { panic!("{e}\n{p:?}"); }
}

/// Pocket wall coincident with the plate boundary (side-face edge split mismatch).
#[test]
fn pocket_on_boundary() { ok(FuzzProgram { w: 65.96744463506025, h: 26.082207520155578, t: 2.6420907625307684, ops: vec![Pocket { x0: 36.10537683592749, y0: 8.717546862201935, w: 29.862067799132756, h: 6.4789332607200425, depth: 2.6185135812770683 }] }); }

/// Boss floating over a pocket: two lumps → two shells.
#[test]
fn floating_boss_two_shells() { ok(FuzzProgram { w: 82.51339241393197, h: 31.40422890937523, t: 13.240913204375072, ops: vec![Pocket { x0: 2.5072399355354955, y0: 5.663351231576942, w: 22.283527092869363, h: 9.787408383084546, depth: 8.373514631042672 }, Boss { x: 7.067480860119684, y: 10.313915251433542, r: 3.258922564378725, height: 10.162720010070286 }] }); }

/// Circle intersection 0.003 mm from a half-arc split point (approximate arc winding).
#[test]
fn near_split_point_intersection() { ok(FuzzProgram { w: 43.03742995246692, h: 64.67027945759693, t: 12.009579100927692, ops: vec![Hole { x: 27.56564827986961, y: 33.535329338046246, r: 3.807304437984483, depth: 15.190947037910734 }, Boss { x: 28.384104804731503, y: 36.3237784182335, r: 3.4078112528626505, height: 9.665727575403894 }] }); }

/// Classification point exactly on a half-arc chord (±π-ambiguous chord winding).
#[test]
fn point_on_arc_chord() { ok(FuzzProgram { w: 62.088404145990815, h: 49.55172900005299, t: 11.013928713329431, ops: vec![Pocket { x0: 28.500091701889797, y0: 15.074472353006197, w: 24.67828835458145, h: 21.68083182315403, depth: 9.331049042128228 }, Boss { x: 52.08609440363021, y: 20.954722475864763, r: 4.6308406054645435, height: 10.22027616305719 }] }); }

/// Two holes in one planar face bridging to the same outer vertex.
#[test]
fn two_holes_shared_bridge() { ok(FuzzProgram { w: 88.60681773986786, h: 72.66859533267692, t: 9.865589704976498, ops: vec![Pocket { x0: 32.10154362636688, y0: 4.180325865231549, w: 38.2786753697473, h: 27.70362431355527, depth: 8.312829154341362 }, Boss { x: 35.56596988320404, y: 43.17296887665622, r: 3.6969798118321173, height: 6.554157305372412 }] }); }

/// Three holes: bridge must attach to the right copy of a duplicated bridge vertex.
#[test]
fn three_holes_duplicate_bridge_vertex() { ok(FuzzProgram { w: 28.560816187498812, h: 20.59895656028832, t: 19.409651817860503, ops: vec![Hole { x: 8.806809590898084, y: 9.670006044986517, r: 1.960893000655968, depth: 24.518846709636794 }, Pocket { x0: 13.419367289195623, y0: 7.9368049274666586, w: 9.903369857323641, h: 3.8234041338812763, depth: 13.06970491620879 }, Hole { x: 13.143875550599498, y: 15.425746983508375, r: 0.6590171634875072, depth: 17.96294870684931 }] }); }

// ---- found by the metamorphic + degenerate-placement fuzzer ----

use agcad_conformance::{build_checked, check_metamorphic};

fn rejected(p: FuzzProgram) {
    let e = build_checked(&p).expect_err("tangent contact must be rejected explicitly");
    assert!(e.contains("tangent"), "{e}");
}

/// Boss centred on the plate edge: circle meets the edge at its half-arc split points.
#[test]
fn boss_centred_on_wall() { ok(FuzzProgram { w: 97.06453798152448, h: 77.71551829756761, t: 18.83629099779418, ops: vec![Boss { x: 55.02580322656405, y: 77.71551829756761, r: 8.445240923853131, height: 6.098140005636715 }] }); }

/// Externally tangent holes pinch material to zero thickness: explicit rejection.
#[test]
fn tangent_holes_rejected() { rejected(FuzzProgram { w: 65.4457949384317, h: 63.38967873894249, t: 8.79456578639114, ops: vec![Hole { x: 35.92767766985629, y: 9.68844425529527, r: 8.60049336271396, depth: 6.055316726009899 }, Hole { x: 45.042213767106276, y: 9.68844425529527, r: 0.5140427345360292, depth: 3.714371741730104 }] }); }

/// Tangent operands (2D boolean ill-conditioned there): rejected before the boolean.
#[test]
fn tangent_operands_rejected() { rejected(FuzzProgram { w: 53.10882870362163, h: 71.13080532680972, t: 2.340051287074071, ops: vec![Hole { x: 32.51535306955584, y: 28.430497957023952, r: 1.6602584969256757, depth: 2.7973380210619303 }, Hole { x: 40.70195006062643, y: 28.430497957023952, r: 6.526338494144911, depth: 1.5652453456451814 }] }); }

/// Boss floating inside a hole: two lumps → two STEP solids.
#[test]
fn two_lumps_two_step_solids() { ok(FuzzProgram { w: 77.63571776686547, h: 46.92888315964298, t: 18.7573521393973, ops: vec![Hole { x: 29.30118220984, y: 19.35906478040845, r: 6.460593754518976, depth: 9.659061223022944 }, Boss { x: 29.30118220984, y: 19.35906478040845, r: 2.4223882764178106, height: 7.693364192188999 }] }); }

/// Fillet arcs tangent to their lines: no sliver splits from ~√ε tangent intersections.
#[test]
fn fillet_plate() { ok(FuzzProgram { w: 67.67821262883325, h: 33.666600741549125, t: 6.576745365224491, ops: vec![Fillet { r: 1.140762393914117 }] }); }

/// Boss overhanging a chamfer: collinear pieces must not produce a spurious crossing point.
#[test]
fn chamfer_collinear_overhang() { ok(FuzzProgram { w: 65.3237085795356, h: 46.03331009212171, t: 17.525627938343447, ops: vec![Chamfer { d: 4.010365147438088 }, Boss { x: 62.00817363873088, y: 1.1801837603201393, r: 1.0556780562254036, height: 2.1666711481751477 }] }); }

/// Chamfer cutting through a hole near the corner: the hole opens into the chamfer face.
#[test]
fn chamfer_through_hole() { ok(FuzzProgram { w: 82.47195216534965, h: 28.784906596719686, t: 14.596546996349971, ops: vec![Hole { x: 80.22360601184158, y: 1.2542426954672021, r: 1.1699280743990608, depth: 4.815794278334313 }, Chamfer { d: 2.98000477562691 }] }); }

/// Reversed order passes through a tangent intermediate state; forward order is valid.
#[test]
fn order_with_tangent_intermediate() {
    let p = FuzzProgram { w: 53.51598840458446, h: 24.552713456628027, t: 2.8993631864303424, ops: vec![Pocket { x0: 46.1913221681214, y0: 0.0, w: 7.32466623646306, h: 11.32806799248911, depth: 2.011309060999062 }, Pocket { x0: 35.05113247926738, y0: 4.968972620161435, w: 14.706338882788307, h: 8.079205011231366, depth: 1.5969589250926333 }, Hole { x: 50.961158369310255, y: 9.008575125777117, r: 1.2036870072545627, depth: 2.4744313103550475 }] };
    check_metamorphic(&p).unwrap();
    ok(p);
}

/// The kernel reports tangency as an actionable TOPOLOGY_FAILED, not an invalid body.
#[test]
fn kernel_rejects_tangent_boolean_with_explanation() {
    use agcad_geo::builders::Profile;
    use agcad_geo::kernel::{ExactKernel, NativeKernel, Solid};
    use agcad_geo::layered::LayeredSolid;
    use agcad_geo::math::Frame;
    use agcad_geo::region2d::BoolOp2;
    let plate = Solid::Layered(LayeredSolid::extrusion(Frame::WORLD, &Profile::rectangle(0.0, 0.0, 50.0, 50.0), 0.0, 10.0));
    let a = NativeKernel.boolean(&plate, &Solid::Layered(LayeredSolid::extrusion(Frame::WORLD, &Profile::circle(20.0, 25.0, 5.0), -1.0, 11.0)), BoolOp2::Difference).unwrap();
    let e = NativeKernel.boolean(&a, &Solid::Layered(LayeredSolid::extrusion(Frame::WORLD, &Profile::circle(28.0, 25.0, 3.0), -1.0, 11.0)), BoolOp2::Difference).unwrap_err();
    assert_eq!(e.status, agcad_driver::AgStatus::TopologyFailed);
    assert!(e.message.contains("tangentially at (25.000000, 25.000000"), "{}", e.message);
}

/// Boss sealing a blind hole: an internal cavity. STEP must carry it as a BREP_WITH_VOIDS void
/// (found by the OCCT oracle: the void had been exported as a second positive solid).
#[test]
fn sealed_cavity_is_a_step_void() {
    let p = FuzzProgram { w: 68.01932048348692, h: 58.52332761632886, t: 13.38121603834861, ops: vec![Hole { x: 56.37971648337965, y: 35.80468098517085, r: 2.50052249024535, depth: 10.728247763577613 }, Boss { x: 56.37971648337965, y: 35.80468098517085, r: 3.7433306589658173, height: 5.881904616810559 }] };
    ok(p.clone());
    let b = agcad_conformance::build(&p).to_brep();
    let (txt, _) = agcad_geo::step_write::write_step(&b, &Default::default());
    assert!(txt.contains("BREP_WITH_VOIDS(") && txt.contains("ORIENTED_CLOSED_SHELL('',*,#") && !txt.contains("MANIFOLD_SOLID_BREP("), "{}", &txt[txt.len().saturating_sub(2000)..]);
    let (bodies, _) = agcad_geo::step_read::read_step(&txt, &Default::default()).unwrap();
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0].shells.len(), 2);
}
