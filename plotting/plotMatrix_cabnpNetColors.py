import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scaledColorMap import scaledColorMap

def plotMatrix_cabnpNetColors(ax,mat,positions=['left'],buffer=1,vmin=None,vmax=None,aspect=1,parcType='cortex',vertices=False,cmap='scaledBlueRed',customParc=None):

    networksList = ['Visual1', 'Visual2', 'Somatomotor', 'Cingulo-Opercular', 'Dorsal-Attention', 'Language', 'Frontoparietal', 'Auditory', 'Default', 'Posterior-Multimodal', 'Ventral-Multimodal', 'Orbito-Affective']
    netColors = {'Visual1': np.array([0., 0., 1.]), 'Visual2': np.array([0.392, 0., 1.]), 'Somatomotor': np.array([0., 1., 1.]), 'Cingulo-Opercular': np.array([0.6, 0., 0.6]), 'Dorsal-Attention': np.array([0., 1., 0.]), 'Language': np.array([0., 0.608, 0.608]), 'Frontoparietal': np.array([1., 1., 0.]), 'Auditory': np.array([0.98, 0.243, 0.984]), 'Default': np.array([1., 0., 0.]), 'Posterior-Multimodal': np.array([0.694, 0.349, 0.157]), 'Ventral-Multimodal': np.array([1., 0.616, 0.]), 'Orbito-Affective': np.array([0.255, 0.49, 0.])}
    
    if parcType == 'cortex':
        netNodeLimit = {'Visual1': np.array([0, 5]), 'Visual2': np.array([6, 59]), 'Somatomotor': np.array([60, 98]), 'Cingulo-Opercular': np.array([ 99, 154]), 'Dorsal-Attention': np.array([155, 177]), 'Language': np.array([178, 200]), 'Frontoparietal': np.array([201, 250]), 'Auditory': np.array([251, 265]), 'Default': np.array([266, 342]), 'Posterior-Multimodal': np.array([343, 349]), 'Ventral-Multimodal': np.array([350, 353]), 'Orbito-Affective': np.array([354, 359])}
        #nNodes = 360
        if vertices:
            netNodeLimit = {'Visual1': np.array([0,2147]), 'Visual2': np.array([2148,8934]), 'Somatomotor': np.array([8935,19041]), 'Cingulo-Opercular': np.array([19042,28143]), 'Dorsal-Attention': np.array([28144,31950]), 'Language': np.array([31951,35181]), 'Frontoparietal': np.array([35182,43403]), 'Auditory': np.array([43404,45036]), 'Default': np.array([45037,56573]), 'Posterior-Multimodal': np.array([56574,57527]), 'Ventral-Multimodal': np.array([57528,58693]), 'Orbito-Affective': np.array([58694,59411])}
            #nNodes = 59412
    elif parcType == 'cortex_subcortex':
        netNodeLimit = {'Visual1': np.array([0, 68]), 'Visual2': np.array([69, 151]), 'Somatomotor': np.array([152, 218]), 'Cingulo-Opercular': np.array([219, 313]), 'Dorsal-Attention': np.array([314, 360]), 'Language': np.array([361, 397]), 'Frontoparietal': np.array([398, 495]), 'Auditory': np.array([496, 541]), 'Default': np.array([542, 650]), 'Posterior-Multimodal': np.array([651, 686]), 'Ventral-Multimodal': np.array([687, 694]), 'Orbito-Affective': np.array([695, 717])}
        #nNodes = 718
        if vertices:
            netNodeLimit = {'Visual1': np.array([0,5167]), 'Visual2': np.array([5168,13838]), 'Somatomotor': np.array([13839,27678]), 'Cingulo-Opercular': np.array([27679,42128]), 'Dorsal-Attention': np.array([42129,48972]), 'Language': np.array([48973,52739]), 'Frontoparietal': np.array([52740,69531]), 'Auditory': np.array([69532,71429]), 'Default': np.array([71430,87099]), 'Posterior-Multimodal': np.array([87100,89154]), 'Ventral-Multimodal': np.array([89155,90336]), 'Orbito-Affective': np.array([90337,91281])}
            #nNodes = 91282
    if not customParc is None:
        netNodeLimit = customParc
        #nNodes = netNodeLimit[list(netNodeLimit.keys())[-1]][1] + 1
    
    if len(positions)>0:
        if (positions[0]=='left') or (positions[0]=='right'):
            nNodes = mat.shape[0]
        if (positions[0]=='top') or (positions[0]=='bottom'):
            nNodes = mat.shape[1]
    
    if vmin==None: 
        vmin = np.nanmin(mat)
    if vmax==None:
        vmax = np.nanmax(mat)
        
    if cmap=='scaledBlueRed':
        cdict = scaledColorMap(vmin,vmax)
        cmap = colors.LinearSegmentedColormap('scaledBlueRed',cdict)
    
    bufferedMat = mat.copy()
    for pos in positions:
        if pos == 'left':
            bufferedMat = np.hstack((np.zeros((bufferedMat.shape[0],buffer)),bufferedMat))
        elif pos == 'right':
            bufferedMat = np.hstack((bufferedMat,np.zeros((bufferedMat.shape[0],buffer))))
        elif pos == 'top':
            bufferedMat = np.vstack((np.zeros((buffer,bufferedMat.shape[1])),bufferedMat))
        elif pos == 'bottom':
            bufferedMat = np.vstack((bufferedMat,np.zeros((buffer,bufferedMat.shape[1]))))
        
    im = ax.imshow(bufferedMat,cmap=cmap,vmin=vmin,vmax=vmax,aspect=aspect,interpolation='nearest')
    
    for net in networksList:
        limits = netNodeLimit[net]
        for pos in positions:
            if (pos=='left') or (pos=='right'):
                width = buffer; height = limits[1] - limits[0] + 1
                xStart = -.5; yStart = -.5 + limits[0]
                if pos=='right':
                    xStart = xStart + nNodes
            elif (pos=='top') or (pos=='bottom'):
                width = limits[1] - limits[0] + 1; height = buffer
                xStart = -.5 + limits[0]; yStart = -.5
                if pos=='bottom':
                    yStart = yStart + nNodes
                
            rect = patches.Rectangle((xStart, yStart), width=width, height=height, color=netColors[net], linewidth=0)
            ax.add_patch(rect)
    
    return ax,im
