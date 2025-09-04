import numpy as np

def scaledColorMap(vMin,vMax):
    
    if vMin > 0:
        vMin = 0
    if vMax < 0:
        vMax = 0
    vRange = vMax - vMin
    maxAbs = np.amax([abs(vMin),abs(vMax)])/vRange
    
    blueLim = np.array([0,0,.25])
    redLim = np.array([.25,0,0])
    if abs(vMax) > abs(vMin):
        vMinRatio = vMin/vMax
        blueLim[0] = 1+4*vMinRatio
        blueLim[1] = 1+1*vMinRatio
        blueLim[2] = 1.75+1.5*vMinRatio
        blueLim[blueLim<0] = 0
        blueLim[blueLim>1] = 1
        
    elif abs(vMax) < abs(vMin):
        vMaxRatio = vMax/vMin
        redLim[2] = 1+2*vMaxRatio
        redLim[1] = 1+2*vMaxRatio
        redLim[0] = 1.75+1.5*vMaxRatio
        redLim[redLim<0] = 0
        redLim[redLim>1] = 1
    
    anchorColors = {-2:[0,.5,1],-1:[0,.75,1],0:[1,1,1],1:[1,.5,.5],2:[1,0,0]}
    
    cdict = {'red':[],'green':[],'blue':[]}
    cdict['red'].append((0,blueLim[0],blueLim[0]))
    cdict['green'].append((0,blueLim[1],blueLim[1]))
    cdict['blue'].append((0,blueLim[2],blueLim[2]))
    for a in np.arange(-2,3):
        anchor = a*maxAbs/4 + (-vMin/vRange)
        if (anchor>0) and (anchor<1):
            cdict['red'].append((anchor,anchorColors[a][0],anchorColors[a][0]))
            cdict['green'].append((anchor,anchorColors[a][1],anchorColors[a][1]))
            cdict['blue'].append((anchor,anchorColors[a][2],anchorColors[a][2]))
    cdict['red'].append((1,redLim[0],redLim[0]))
    cdict['green'].append((1,redLim[1],redLim[1]))
    cdict['blue'].append((1,redLim[2],redLim[2]))
    
    return cdict
