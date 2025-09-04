import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import matplotlib.colors as colors
from scaledColorMap import scaledColorMap

cabnpDir = f'/projects/f_mc1689_1/AnalysisTools/ColeAnticevicNetPartition'
vertexFile = f'{cabnpDir}/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'
vertexLabels = np.squeeze(nib.load(vertexFile).get_fdata())
lh = f'{cabnpDir}/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii'
rh = f'{cabnpDir}/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii'
LR = np.full((len(vertexLabels)),0)
for n in range(360):
    idx = np.where(vertexLabels==(n+1))[0]
    if (n<180):
        LR[idx] = -1#'L'
    elif (n>=180):
        LR[idx] = 1
LRbound = 32492
medWall = nib.load(f'{cabnpDir}/Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii').get_fdata().squeeze()
sulc = nib.load(f'{cabnpDir}/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii').get_fdata().squeeze()
sulcFull = np.full((medWall.shape[0]),np.nan)
sulcFull[medWall==0] = sulc
sulcFull[medWall!=0] = -1
sulcFull = (sulcFull-np.nanmin(sulcFull))/(np.nanmax(sulcFull)-np.nanmin(sulcFull))
sulcFull = 1-sulcFull
del sulc

def parcelsToVertices(pData):
    nNodes = pData.shape[0]
    sz = np.array(pData.shape)
    sz[0] = len(vertexLabels)
    vData = np.full(tuple(sz),np.nan)
    for n in range(nNodes):
        idx = np.where(vertexLabels==(n+1))[0]
        vData[idx] = pData[n]
    return vData

def plotToCortex(data,vertices=False,vmin=None,vmax=None,cmap='scaledBlueRed',vertLRSep=False):
    # data - parcel-/vertex-wise values, in original MSMall order, not reordered by networks
    
    if not vertices:
        vData = parcelsToVertices(data)
    else:    
        if not vertLRSep:
            vData = data
        else:
            vData = np.zeros((len(vertexLabels)))
            vData[LR==-1] = data[:,0]
            vData[LR==1] = data[:,1]
        
    vDataFull = np.full((medWall.shape[0]),np.nan)
    vDataFull[medWall==0] = vData[LR!=0]
    del vData
    
    if vmin is None:
        vmin = np.nanmin(vDataFull)
        vmax = np.nanmax(vDataFull)
    if cmap=='scaledBlueRed':
        cdict = scaledColorMap(vmin,vmax)
        cmap = colors.LinearSegmentedColormap('scaledBlueRed',cdict)

    h = 4
    w = 8
    fig = plt.figure(figsize=(w,h))
    sc = .85
    axs = fig.add_subplot(1,1,1,projection='3d',position=[-.03125,.4375+.0075,.3125,.625])
    fig = plotting.plot_surf(lh,surf_map=vDataFull[0:LRbound],bg_map=sulcFull[0:LRbound],darkness=.25,hemi='left',view='dorsal',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    fig = plotting.plot_surf(rh,surf_map=vDataFull[LRbound:],bg_map=sulcFull[LRbound:],darkness=.25,hemi='right',view='dorsal',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.view_init(elev=90,azim=0,roll=90)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)
    axs = fig.add_subplot(1,1,1,projection='3d',position=[-.03125,-.0625-.0075,.3125,.625])
    fig = plotting.plot_surf(lh,surf_map=vDataFull[0:LRbound],bg_map=sulcFull[0:LRbound],darkness=.25,hemi='left',view='ventral',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    fig = plotting.plot_surf(rh,surf_map=vDataFull[LRbound:],bg_map=sulcFull[LRbound:],darkness=.25,hemi='right',view='ventral',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.view_init(elev=-90,azim=0,roll=90)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)

    axs = fig.add_subplot(1,1,1,projection='3d',position=[.25,.5,.25,.5])
    fig = plotting.plot_surf(lh,surf_map=vDataFull[0:LRbound],bg_map=sulcFull[0:LRbound],darkness=.25,hemi='left',view='lateral',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)
    axs = fig.add_subplot(1,1,1,projection='3d',position=[.5,.5,.25,.5])
    fig = plotting.plot_surf(rh,surf_map=vDataFull[LRbound:],bg_map=sulcFull[LRbound:],darkness=.25,hemi='right',view='lateral',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)
    axs = fig.add_subplot(1,1,1,projection='3d',position=[.25,0,.25,.5])
    fig = plotting.plot_surf(lh,surf_map=vDataFull[0:LRbound],bg_map=sulcFull[0:LRbound],darkness=.25,hemi='left',view='medial',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)
    axs = fig.add_subplot(1,1,1,projection='3d',position=[.5,0,.25,.5])
    fig = plotting.plot_surf(rh,surf_map=vDataFull[LRbound:],bg_map=sulcFull[LRbound:],darkness=.25,hemi='right',view='medial',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)

    axs = fig.add_subplot(1,1,1,projection='3d',position=[.75,.5,.25,.5])
    fig = plotting.plot_surf(lh,surf_map=vDataFull[0:LRbound],bg_map=sulcFull[0:LRbound],darkness=.25,hemi='left',view='anterior',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    fig = plotting.plot_surf(rh,surf_map=vDataFull[LRbound:],bg_map=sulcFull[LRbound:],darkness=.25,hemi='right',view='anterior',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)
    axs = fig.add_subplot(1,1,1,projection='3d',position=[.75,0,.25,.5])
    fig = plotting.plot_surf(lh,surf_map=vDataFull[0:LRbound],bg_map=sulcFull[0:LRbound],darkness=.25,hemi='left',view='posterior',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    fig = plotting.plot_surf(rh,surf_map=vDataFull[LRbound:],bg_map=sulcFull[LRbound:],darkness=.25,hemi='right',view='posterior',axes=axs,figure=fig,cmap=cmap,bg_on_data=True,vmin=vmin,vmax=vmax)
    axs.set_xlim(-88.6*sc,88.6*sc); axs.set_ylim(-110.6*sc,66.6*sc); axs.set_zlim(-40.9*sc,80.9*sc)

    axs = fig.add_subplot(1,1,1,position=[.4,.06,.2,.01])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=colors.Normalize(vmin=vmin,vmax=vmax)),cax=axs,orientation='horizontal')
    cb.outline.set_visible(False)

    return fig