"""FADE Python adaptation of the LIVE 2015 MATLAB release.
Original source: https://live.ece.utexas.edu/research/fog/FADE_release.zip
Small fitted statistical parameters are used without retraining.

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2015 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
at The University of Texas at Austin (UT Austin,http://www.utexas.edu), 
is acknowledged in any publication that reports research using this code. The research
is to be cited in the bibliography as:

1. L. K. Choi, J. You, and A. C. Bovik, "Referenceless Prediction of Perceptual Fog Density and Perceptual Image Defogging,"
IEEE Transactions on Image Processing, to appear (2015).
2. L. K. Choi, J. You, and A. C. Bovik, "Referenceless perceptual fog density prediction model," in Proc. SPIE Human Vis. Electron. Imag.,
Feb. 2014, 90140H.
3. L. K. Choi, J. You, and A. C. Bovik, "FADE Software Release," 
URL: http://live.ece.utexas.edu/research/fog/FADE_release.zip, 2015

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%
"""
from pathlib import Path
import warnings
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.signal import convolve2d
from scipy import ndimage as ndi

MODEL_DIR=Path(__file__).resolve().parent/'models_cache'/'FADE'
CONFIG=dict(dark_patch=15,omega=0.95,t_min=0.40,strength=0.95,
            fade_start=0.4,fade_end=1.2,dark_start=0.04,dark_end=0.18,
            cv_start=0.12,cv_end=0.45,texture_start=0.008,texture_end=0.035,
            sky_start=0.78,sky_end=0.96,mask_sigma=5.0,
            source_sigma=70.0,source_start=0.015,source_end=0.12,
            guided_radius=24,guided_eps=0.0025,max_luminance_drop=0.25,
            sky_candidate_y=0.82,sky_candidate_chroma=0.15,
            sky_min_fraction=0.005,sky_reference_max_clipping=0.10,
            sky_gain_min=0.65,sky_gain_sigma=18.0,
            sky_target_quantile=0.25,sky_target_offset=0.03,sky_detail_gamma=1.50,
            sky_match_fade_start=0.65,sky_match_fade_end=0.85)

def blocks(a):
    h,w=a.shape
    return a.reshape(h//8,8,w//8,8).transpose(0,2,1,3).reshape(-1,64)

def patch_var(a):
    count=np.isfinite(a).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        v=np.nanvar(a,axis=1,ddof=1)
    v[count==1]=0
    return v

def contrast_energy(rgb):
    r,g,b=rgb.transpose(2,0,1)
    sigma=3.25;x=np.arange(-3*sigma,3*sigma+0.01,1)
    gauss=np.exp(-x*x/(2*sigma*sigma));gauss/=gauss.sum()
    kernel=(x*x/sigma**4-1/sigma**2)*gauss
    kernel-=kernel.mean();kernel/=np.sum(.5*x*x*kernel)
    result=[]
    for a,threshold in zip((.299*r+.587*g+.114*b,.5*r+.5*g-b,r-g),
                           (.0009225496406318721,.0008969246659629488,.0002069284034165411)):
        # Match the source release's copied (not mirrored) ten-pixel border.
        padded=np.concatenate((a[:10],a,a[-10:]),axis=0)
        padded=np.concatenate((padded[:,:10],padded,padded[:,-10:]),axis=1)
        energy=np.hypot(convolve2d(padded,kernel[None,:],mode='same'),
                        convolve2d(padded,kernel[:,None],mode='same'))[10:-10,10:-10]
        maximum=energy.max()
        response=energy*maximum/np.maximum(energy+maximum*.1,1e-12)-threshold*255
        result.append(np.where(response>1e-7,response,0))
    return result

def fade(rgb8):
    """Twelve original patch features and original fitted MVGs; no neural model."""
    h,w=rgb8.shape[:2];h-=h%8;w-=w%8
    rgb=rgb8[:h,:w].astype(np.float64)
    r,g,b=rgb.transpose(2,0,1)
    gray=np.floor(.298936021293775*r+.587043074451121*g+.114020904255103*b+.5)
    gaussian=cv2.getGaussianKernel(7,7/6)
    mu=cv2.sepFilter2D(gray,-1,gaussian,gaussian,borderType=cv2.BORDER_REPLICATE)
    std=np.sqrt(np.abs(cv2.sepFilter2D(gray*gray,-1,gaussian,gaussian,borderType=cv2.BORDER_REPLICATE)-mu*mu))
    mscn=(gray-mu)/(std+1)
    with np.errstate(divide='ignore',invalid='ignore'):
        cv=std/mu
    pair=blocks(mscn*np.roll(mscn,1,axis=0))
    left=np.where(pair<=0,pair,np.nan);right=np.where(pair>=0,pair,np.nan)
    energy=contrast_energy(rgb)
    gray_blocks=blocks(gray).astype(np.int32)
    hist=np.zeros((len(gray_blocks),256),np.float64)
    np.add.at(hist,(np.repeat(np.arange(len(hist)),64),gray_blocks.ravel()),1)
    hist/=64
    entropy=-np.sum(hist*np.log2(np.maximum(hist,1e-12)),axis=1)
    maximum=rgb.max(axis=2);minimum=rgb.min(axis=2)
    sat=(maximum-minimum)/np.maximum(maximum,1e-12)
    rg,by=blocks(r-g),blocks(.5*(r+g)-b)
    colorfulness=np.sqrt(rg.var(axis=1,ddof=1)+by.var(axis=1,ddof=1))+.3*np.hypot(rg.mean(axis=1),by.mean(axis=1))
    features=np.stack((patch_var(blocks(mscn)),patch_var(right),patch_var(left),
                      blocks(std).mean(axis=1),blocks(cv).mean(axis=1),
                      *[blocks(e).mean(axis=1) for e in energy],entropy,
                      blocks(minimum/255).mean(axis=1),blocks(sat).mean(axis=1),colorfulness),axis=1)
    features=np.log1p(features)
    valid=np.isfinite(features).all(axis=1)
    variance=features[valid].var(axis=1,ddof=1)
    distances=[]
    for word in ('fogfree','foggy'):
        model=loadmat(MODEL_DIR/f'natural_{word}_image_features_ps8.mat')
        mu_model=model[f'mu_{word}param'].ravel();cov=model[f'cov_{word}param']
        diff=mu_model-features[valid]
        # Faithful to release: per-patch variance is a scalar broadcast to 12x12.
        matrix=(cov[None,:,:]+variance[:,None,None])/2
        inverse=np.linalg.pinv(matrix)
        distance=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',diff,inverse,diff),0))
        full=np.full(len(features),np.nan);full[valid]=distance
        distances.append(full)
    df,dff=distances
    score=float(np.nanmean(df)/(np.nanmean(dff)+1))
    density=(df/(dff+1)).reshape(h//8,w//8)
    return score,density

def smoothstep(a,lo,hi):
    t=np.clip((a-lo)/(hi-lo),0,1)
    return t*t*(3-2*t)

def guided(guide,value,radius,eps):
    def mean(a):return cv2.boxFilter(a,-1,(radius*2+1,radius*2+1),borderType=cv2.BORDER_REFLECT)
    mi,mp=mean(guide),mean(value)
    a=(mean(guide*value)-mi*mp)/(np.maximum(mean(guide*guide)-mi*mi,0)+eps)
    return mean(a)*guide+mean(mp-a*mi)

def luminance(rgb):
    return np.einsum('...c,c->...',rgb,np.array([.299,.587,.114],np.float32))

def analyze(rgb8):
    """No scene coordinates: weak haze evidence + retained texture, throughout image."""
    rgb=rgb8.astype(np.float32)/255;y=luminance(rgb)
    density,map8=fade(rgb8)
    fog=cv2.resize(np.nan_to_num(map8,nan=0).astype(np.float32),(y.shape[1],y.shape[0]),interpolation=cv2.INTER_LINEAR)
    dark=cv2.erode(rgb.min(axis=2),np.ones((CONFIG['dark_patch'],)*2,np.uint8))
    mu=cv2.GaussianBlur(y,(0,0),7)
    std=np.sqrt(np.maximum(cv2.GaussianBlur(y*y,(0,0),7)-mu*mu,0))
    cv=std/np.maximum(mu,.03)
    score=smoothstep(fog,CONFIG['fade_start'],CONFIG['fade_end'])
    score*=smoothstep(dark,CONFIG['dark_start'],CONFIG['dark_end'])
    score*=1-smoothstep(cv,CONFIG['cv_start'],CONFIG['cv_end'])
    score*=smoothstep(std,CONFIG['texture_start'],CONFIG['texture_end'])
    # A light-source neighborhood helps avoid confusing uniformly gray road with glare.
    source=cv2.GaussianBlur(smoothstep(y,.85,.98),(0,0),CONFIG['source_sigma'])
    score*=smoothstep(source,CONFIG['source_start'],CONFIG['source_end'])
    score*=1-smoothstep(y,CONFIG['sky_start'],CONFIG['sky_end'])
    score=cv2.GaussianBlur(score,(0,0),CONFIG['mask_sigma'])
    score=np.clip(guided(y,score,CONFIG['guided_radius'],CONFIG['guided_eps']),0,1)
    score*=1-smoothstep(y,CONFIG['sky_start'],CONFIG['sky_end'])
    return dict(fade=density,fade_map=fog,dark=dark,local_std=std,cv=cv,confidence=score,source_field=source)

def balance_sky_brightness(rgb8,analysis_shape):
    """Match bright neutral components to the largest reliable unsaturated component.

    This changes rendered brightness, including clipped white pixels; no detail is
    invented. All component selection is photometric, without left/right coordinates.
    """
    h,w=analysis_shape
    small=cv2.resize(rgb8,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    y=luminance(small)
    neutral=(small.max(axis=2)-small.min(axis=2))<CONFIG['sky_candidate_chroma']
    mask=(y>CONFIG['sky_candidate_y'])&neutral
    mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8)).astype(bool)
    labels,n=ndi.label(mask)
    regions=[]
    for r in range(1,n+1):
        m=labels==r
        if m.sum()<CONFIG['sky_min_fraction']*h*w:continue
        yy,xx=np.where(m)
        regions.append(dict(component=r,area=int(m.sum()),mean_before=float(y[m].mean()),
                            median_before=float(np.median(y[m])),
                            saturated_fraction=float((y[m]>=.995).mean()),
                            centroid_x=float(xx.mean()/w),centroid_y=float(yy.mean()/h)))
    reliable=[r for r in regions if r['saturated_fraction']<CONFIG['sky_reference_max_clipping']]
    if not reliable or len(regions)<2:
        return rgb8,dict(status='skipped_no_reliable_bright_reference',components=regions),np.ones_like(y)
    reference=max(reliable,key=lambda r:r['area'])
    reference_mean=reference['mean_before']
    reference_quantile=float(np.quantile(y[labels==reference['component']],CONFIG['sky_target_quantile']))
    target=max(.65,reference_quantile-CONFIG['sky_target_offset'])
    known=np.zeros_like(y,dtype=bool);samples=np.ones_like(y);exponents=np.ones_like(y)
    for r in regions:
        m=labels==r['component'];known|=m
        compress=r['mean_before']>reference_mean+.02
        gamma=CONFIG['sky_detail_gamma'] if compress else 1.
        component_target=target if compress else r['mean_before']
        gain=float(np.clip(component_target/float((y[m]**gamma).mean()),CONFIG['sky_gain_min'],1))
        samples[m]=gain;r['requested_gain']=gain
        exponents[m]=gamma;r.update(gamma=gamma,target_luminance=component_target)
    # Extend the gain through nearby bright gaps, then smooth only this gain field.
    nearest=ndi.distance_transform_edt(~known,return_distances=False,return_indices=True)
    field=samples[tuple(nearest)]
    field=cv2.GaussianBlur(field,(0,0),CONFIG['sky_gain_sigma'])
    field=np.clip(guided(y,field,24,.01),CONFIG['sky_gain_min'],1)
    gamma_field=exponents[tuple(nearest)]
    gamma_field=cv2.GaussianBlur(gamma_field,(0,0),CONFIG['sky_gain_sigma'])
    gamma_field=np.clip(guided(y,gamma_field,24,.01),1,CONFIG['sky_detail_gamma'])
    hh,ww=rgb8.shape[:2]
    rgb=rgb8.astype(np.float32)/255;native_y=luminance(rgb)
    gain=cv2.resize(field,(ww,hh),interpolation=cv2.INTER_LINEAR)
    gamma=cv2.resize(gamma_field,(ww,hh),interpolation=cv2.INTER_LINEAR)
    gain=np.clip(gain*np.maximum(native_y,1e-5)**(gamma-1),CONFIG['sky_gain_min'],1)
    gate=smoothstep(native_y,CONFIG['sky_match_fade_start'],CONFIG['sky_match_fade_end'])
    gain=1+(gain-1)*gate
    out=np.clip(np.rint(rgb*gain[:,:,None]*255),0,255).astype(np.uint8)
    after=luminance(cv2.resize(out,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255)
    for r in regions:
        m=labels==r['component'];r['mean_after']=float(after[m].mean())
        r['mean_error_before']=abs(r['mean_before']-r['target_luminance'])
        r['mean_error_after']=abs(r['mean_after']-r['target_luminance'])
    return out,dict(status='applied',reference_component=reference['component'],
                    target_luminance=target,reference_mean=reference_mean,
                    reference_quantile=reference_quantile,components=regions,
                    interpretation='Darker highlight rendering with increased slope in retained unsaturated variations; clipped pixels remain flat'),field

def suppress(rgb8,analysis_shape):
    """Restrained dark-channel restoration of the enhanced image, all spatial decisions automatic."""
    h,w=analysis_shape
    small8=cv2.resize(rgb8,(w,h),interpolation=cv2.INTER_AREA)
    small=small8.astype(np.float32)/255;ys=luminance(small)
    info=analyze(small8)
    # Estimate airlight from the brightest pixels among the highest dark-channel samples.
    flat=info['dark'].ravel();n=max(1,int(len(flat)*.001))
    candidates=np.argpartition(flat,-n)[-n:]
    colors=small.reshape(-1,3)[candidates]
    brightest=np.argsort(colors.mean(axis=1))[-max(1,n//10):]
    air=np.clip(colors[brightest].mean(axis=0),.65,1)
    normalized=np.min(small/air,axis=2)
    dark_norm=cv2.erode(normalized,np.ones((CONFIG['dark_patch'],)*2,np.uint8))
    raw=1-CONFIG['omega']*dark_norm
    transmission=np.clip(guided(ys,raw,CONFIG['guided_radius'],CONFIG['guided_eps']),CONFIG['t_min'],1)
    hh,ww=rgb8.shape[:2]
    alpha=cv2.resize(info['confidence']*CONFIG['strength'],(ww,hh),interpolation=cv2.INTER_LINEAR)
    t=cv2.resize(transmission,(ww,hh),interpolation=cv2.INTER_LINEAR)
    rgb=rgb8.astype(np.float32)/255;y=luminance(rgb)
    recovered=(rgb-air)/t[:,:,None]+air
    # Continuous cap prevents large black patches while retaining channel contrast.
    delta=recovered-rgb
    drop=np.maximum(0,y-luminance(recovered))
    alpha*=np.minimum(1,CONFIG['max_luminance_drop']/np.maximum(drop,1e-5))
    alpha*=1-smoothstep(y,CONFIG['sky_start'],CONFIG['sky_end'])
    result=np.clip(rgb+delta*alpha[:,:,None],0,1)
    output=np.clip(np.rint(result*255),0,255).astype(np.uint8)
    info.update(airlight=air.tolist(),transmission=transmission,alpha_small=cv2.resize(alpha,(w,h),interpolation=cv2.INTER_AREA))
    info['before_sky_balance']=output
    output,matching,gain_field=balance_sky_brightness(output,analysis_shape)
    info.update(sky_matching=matching,sky_gain_field=gain_field)
    return output,info
