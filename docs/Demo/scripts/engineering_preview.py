"""Scientific views of the actual GPU quality grid and trimmed concept panel."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/engineering/demo'


def main():
    data=np.load(OUT/'quality-grid.npz');report=json.loads((OUT/'validation.json').read_text())
    fig=plt.figure(figsize=(13,5.8),facecolor='#f5f7fa');ax=fig.add_subplot(121,projection='3d');heat=fig.add_subplot(122)
    norm=Normalize(0,max(max(map(abs,r['principal_curvature_range_per_mm'])) for r in report['quality'].values()))
    theta=np.linspace(0,2*np.pi,24,endpoint=False);hole=np.c_[.55+.15*np.cos(theta),.52+.22*np.sin(theta)]
    for name in ('left','right'):
        grid=data[name+'_grid'];xyz=grid[:,:,:3].copy();zebra=(grid[:,:,41]+1)/2
        if name=='right':
            u,v=np.meshgrid(np.linspace(0,1,len(grid)),np.linspace(0,1,len(grid)),indexing='ij')
            mask=PolygonPath(np.vstack([hole,hole[0]]),closed=True).contains_points(np.c_[u.ravel(),v.ravel()]).reshape(u.shape)
            xyz[mask]=np.nan
        colors=plt.colormaps['Greys'](.15+.7*zebra)
        ax.plot_surface(xyz[:,:,0],xyz[:,:,1],xyz[:,:,2],facecolors=colors,shade=False,rstride=1,cstride=1,linewidth=0,antialiased=False)
        curvature=np.maximum(np.abs(grid[:,:,36]),np.abs(grid[:,:,37]));curvature[np.isnan(xyz[:,:,0])]=np.nan
        heat.pcolormesh(grid[:,:,0],grid[:,:,1],curvature,cmap='viridis',norm=norm,shading='gouraud')
    ax.set_title('GPU reflection stripes / trimmed support',pad=15);ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)')
    ax.set_box_aspect((2,1,.6));ax.view_init(30,-65)
    heat.set_title('Sampled maximum absolute principal curvature');heat.set_xlabel('X (mm)');heat.set_ylabel('Y (mm)');heat.set_aspect('equal')
    fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=heat,shrink=.65,label='1 / mm')
    fig.suptitle('Engineering support demo | two quintic panels + polygonal UV opening',fontsize=14,y=.98)
    fig.text(.04,.05,'Actual CUDA grid. Left: Z display exaggerated; constant view direction.\nSampled diagnostics; no global Class-A certification.',fontsize=9,color='#46536b')
    fig.text(.57,.05,f"G3 jet residual: {max(report['seam']['jet_max_mm']):.2e} mm\nSTEP sampled roundtrip: {report['cad']['deformation']['symmetric_sampled_max_mm']:.2e} mm",fontsize=9,color='#46536b')
    fig.subplots_adjust(top=.85,bottom=.18,left=.03,right=.96,wspace=.25)
    fig.savefig(OUT/'quality-preview.png',dpi=150);plt.close(fig)
    print(str(OUT/'quality-preview.png'))


if __name__=='__main__':main()
