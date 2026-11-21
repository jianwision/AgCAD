"""Render actual resident GPU samples with shared before/after color scales."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/resident-surfaces/demo'


def main():
    data=np.load(OUT/'quality-grid.npz');t=np.linspace(0,1,41);u,v=np.meshgrid(t,t,indexing='ij')
    hole=np.array([[.52+.16*np.cos(a),.55+.24*np.sin(a)] for a in np.linspace(0,2*np.pi,32,endpoint=False)])
    path=PlotPath(np.r_[hole,hole[:1]]);mask=path.contains_points(np.column_stack((u.ravel(),v.ravel()))).reshape(41,41)
    fig=plt.figure(figsize=(15,8.5),layout='constrained');maps=[]
    for mode in ('before','after'):
        images=[]
        for i in range(3):
            image=np.log10(np.maximum(data[f'{mode}_{i}'][:,:,45],1e-18)).T.copy()
            if i==1:image[mask.T]=np.nan
            images.append(image)
        maps.append(np.concatenate(images,axis=1))
    lo=min(float(np.nanmin(a)) for a in maps);hi=max(float(np.nanmax(a)) for a in maps)
    for col,mode in enumerate(('before','after')):
        ax=fig.add_subplot(2,2,col+1,projection='3d')
        for i in range(3):
            grid=data[f'{mode}_{i}'];z=grid[:,:,2].copy();colors=plt.cm.gray(.15+.7*(grid[:,:,41]+1)/2)
            if i==1:z[mask]=np.nan
            ax.plot_surface(grid[:,:,0],grid[:,:,1],z,facecolors=colors,rstride=1,cstride=1,linewidth=0,antialiased=True,shade=False)
        ax.set_title(('Seed surfaces' if mode=='before' else 'Joint G3 optimization')+' — reflection stripes',fontsize=12)
        ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',xlim=(-2,302),ylim=(-2,62),zlim=(-2,65))
        ax.set_box_aspect((304,64,67));ax.view_init(elev=30,azim=-65)
        heat=fig.add_subplot(2,2,col+3)
        im=heat.imshow(maps[col],origin='lower',extent=(0,3,0,1),aspect='auto',vmin=lo,vmax=hi,cmap='magma')
        for x in (1,2):heat.axvline(x,color='white',linewidth=.6,alpha=.7)
        heat.set(title='Mean-curvature gradient squared',xlabel='Patch index + local U',ylabel='Local V')
        fig.colorbar(im,ax=heat,label=r'$\log_{10}(|\nabla_S H|^2\,/\,\mathrm{mm}^{-4})$')
    fig.suptitle('Resident GPU surface modeling: three quintic panels, fixed boundary locks, bounded shape deviation',fontsize=14)
    fig.supxlabel('Actual GPU samples. Shared scales; physical 3D aspect. Sampled quality does not certify global Class-A.',fontsize=10)
    path=OUT/'quality-preview.png';fig.savefig(path,dpi=160);plt.close(fig)
    (OUT/'quality-preview.json').write_text(json.dumps({'source_grid_sha256':hashlib.sha256((OUT/'quality-grid.npz').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'shared_color_scales':True,'physical_3d_aspect':True},indent=2),encoding='utf-8');print(path)


if __name__=='__main__':main()
