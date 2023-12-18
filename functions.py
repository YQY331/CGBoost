import torch.nn.functional as F
import torch
def readTiff(path):
    """
    :param path:
    :return: data,geotrans,proj
    """
    import gdal
    import numpy as np
    dataset=gdal.Open(path)
    width=dataset.RasterXSize
    height=dataset.RasterYSize
    geotrans=dataset.GetGeoTransform()
    proj=dataset.GetProjection()
    data=dataset.ReadAsArray(0,0,width,height)
    #data=np.array(data,dtype=np.float)
    return data,geotrans,proj
def writeTiff(path, im_data, im_geotrans, im_proj):
    import numpy as np
    import gdal,os
    #im_data.astype(np.float)  # Convert arr data type
    # Get data encoding type
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float64
    # Get image channel number and size
    if len(im_data.shape) >= 3:
        im_bands, im_height, im_width = im_data.shape[0], im_data.shape[1], im_data.shape[2]
    elif len(im_data.shape) == 2:
        im_data = im_data[np.newaxis, :, :]
        im_bands, im_height, im_width = im_data.shape[0], im_data.shape[1], im_data.shape[2]
    else:
        im_bands, im_height, im_width = 0, 0, 1
        print("The data shape is only one-dimensional！！！")
        exit()
    # Create a file
    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(path+".ovr"):
        os.remove(path+".ovr")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype, options=["BIGTIFF=IF_NEEDED"])
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # Write affine transformation parameters
        dataset.SetProjection(im_proj)  # Write projection
    else:
        print("Failed to create tif file")
        exit()
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])  # Write band data
    del dataset
def clamp(x):
    import numpy as np
    quatervalue = np.unique(x)
    min, max =quatervalue[int(0.02 * len(quatervalue))],quatervalue[int(0.98* len(quatervalue))]
    if min==0:
        min=1
    if max==len(quatervalue):
        max=len(quatervalue)-1
    x= np.clip(x,min,max)
    return x
def listbyindice(List1,indice):
    tmp=[]
    for index in indice:
        tmp=tmp+[List1[index]]
    return tmp
def readout_nodes(graph,featkey="feat",op="std"):
    import dgl,torch
    torchfunc={"std":torch.std,"mean":torch.mean}
    gs=dgl.unbatch(graph)
    def mapfun(tmg):
        return torchfunc[op](tmg.ndata[featkey],dim=0)
    # from multiprocessing.dummy import Pool
    # import multiprocessing as mp
    # pool=Pool(mp.cpu_count()-2)
    onefeat=torch.stack(list(map(mapfun,gs)),dim=0)
    # pool.close()
    return onefeat





