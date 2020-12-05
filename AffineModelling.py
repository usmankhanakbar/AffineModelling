import numpy as np
import os
import pydicom
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as ElementWiseKernel
from pycuda.cumath import sqrt
from pycuda.gpuarray import min
from pycuda.gpuarray import max
from pycuda.gpuarray import sum
import mayavi.mlab as mlab
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import mutual_info_score as mis
from sklearn.metrics import normalized_mutual_info_score as nmis
#import vtk.vtkPLYWriter as vply


#from codepy.cgen import *
#from codepy.bpl import BoostPythonModule
#from codepy.cuda import CudaModule
###########################################################
#canvas = sc.SceneCanvas(keys='interactive')
#view = canvas.central_widget.add_view()

print("[INFO] loading images...")
#imagePath = r"E:\Thesis\Databases\l3"
imagePath = r"E:\Thesis\Databases\c1"
pixel_data = []
image_position = []     
image_orientation = [] 
slice_thickness = []
count = 1
for dirName, subdirList, fileList in os.walk(imagePath):
    for filename in fileList:
        if ".dcm" in filename.lower():
            print(filename)
            image = pydicom.dcmread(dirName+"\\"+filename,force = True)
            image.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian 
            pixel_data.append(image.pixel_array)
            image_position.append(image.ImagePositionPatient)
            image_orientation.append(image.ImageOrientationPatient)
            slice_thickness.append(image.SliceThickness)
        else:
            continue
array_pix=np.array(pixel_data)

array_pix_shape = array_pix.shape # in order to obtain size of Px , Py and Pz arrays
P_array_len = array_pix_shape[0]*array_pix_shape[1]*array_pix_shape[2] 
###################################################
Px = np.zeros(array_pix_shape,dtype = float)
Px = np.float32(Px)
Py = np.zeros(array_pix_shape,dtype = float)
Py = np.float32(Py)
Pz = np.zeros(array_pix_shape,dtype = float)
Pz = np.float32(Pz)
###################################################
orientation_of_slice = np.array(image_orientation)
Xxyz = orientation_of_slice[0][0:3]
Yxyz = orientation_of_slice[0][3:6]
Sxyz = np.array(image_position)  # Position of SLice of the Dicom Affine Matrix
pixel_spacing = image.PixelSpacing
DeltaX = pixel_spacing[0] # Although we are catering for pixel spacing 
DeltaY = pixel_spacing[1] # Slice Spacing is missing in this formulation
############################################################################33
S_array = np.float32(np.array(image_position))
_delta_t1 = S_array[0][0]-S_array[len(S_array)-1][0]
_delta_t2 = S_array[1][1]-S_array[len(S_array)-1][1]
_delta_t3 = S_array[2][2]-S_array[len(S_array)-1][2]
##########################################################################
# For the slice formulation I foresee that I need to keep the z coordinate constant
# As I keep z coordinate constant and then I scan for X-Y coordinates plotting each scan
# Then I give slice spacing and plot for the next slice i-e I need to shift the Z coordinate 
# through slice thickness

#######################################################################
Xxyz_32 = Xxyz.astype(np.float32)
Yxyz_32 = Yxyz.astype(np.float32)
Sxyz_32 = Sxyz[0].astype(np.float32)   
Sxyz = np.float32(Sxyz.reshape(Sxyz.T.shape))
_Sxyz = []
for i in range(len(Sxyz[0])):
    _Sxyz = np.append(_Sxyz,np.append(Sxyz[:,[i]],1))
    
_Sxyz = _Sxyz.reshape([Sxyz.shape[1],4])
DeltaX_32 = np.float32(DeltaX)
DeltaY_32 = np.float32(DeltaY)

X_array = Xxyz_32*DeltaX_32
Y_array = Yxyz_32*DeltaY_32
X_array = np.pad(X_array,(0,1),'constant')
Y_array = np.pad(Y_array, (0, 1), 'constant')
X_array = X_array.reshape(4,1)
Y_array = Y_array.reshape(4,1)
zero_vec = np.float32(np.zeros([4,1],dtype=float))
P = []
_array_3d = []
_Ones = np.float32(np.ones([1,array_pix.shape[1]])) # For multiplication with indices
_indices = np.float32(range(array_pix.shape[1])) # indices of the range(512)
row_index = np.reshape(np.float32(np.array(range(array_pix.shape[1]))),[1,array_pix.shape[1]])
col_index = np.reshape(np.float32(np.array(range(array_pix.shape[1]))),[1,array_pix.shape[1]])
zero_index = np.float32(np.zeros([1,array_pix.shape[1]]))
one_index = np.float32(np.ones([1,array_pix.shape[1]]))
################################################
_Delta_vec = np.float32(np.array([_delta_t1/(-46),_delta_t2/(-46),_delta_t3/(-46),0]))
_Delta_vec = _Delta_vec.T
##############################################3
for j in range(array_pix.shape[0]):
    for i in range(array_pix.shape[1]):
        Sxyz_array = _Sxyz[j][:]
        Affine_Matrix = np.c_[X_array,Y_array,_Delta_vec,Sxyz_array] # I have excluded zero_vec
        row_indices = i * _Ones
        Index_array = np.r_[row_indices,col_index,zero_index,one_index]
        P_array = Affine_Matrix.dot(Index_array)
        # Now through this we access each pix of the 2D 
        array_pix[j,row_indices.astype(int),col_index.astype(int)]
        # _array[j,P_array.shape[0],i,col_index.astype(int)] =
        #head = Node.construct(array_pix[j], 0, 0, 512, 512) 
        
        P.append(np.concatenate((P_array,array_pix[j,row_indices.astype(int),col_index.astype(int)]),axis=0))
P = np.float32(P)
#################################################################################
# This means that P[48639,:,:] == np.concatenate((P_array,array_pix[j,row_indices.astype(int),col_index.astype(int)]),axis=0)
# Which means that for each x,y,z coordinates of RCS the fourth value is for the pixel i-e first 3 values are coordinates 
# and fourth is the pixel value
# for i in range(512*95)
# P[i ,0:3,:] is the 3D coordinates for each of the 1 x 3 x 512 pixels for each 3d coordinates
###############################################################################
x = []
y = []
z = []
pix = []

for i in range (np.int(P.shape[0])):
    for j in range(array_pix.shape[1]):
        x.append(P[i,0,j])
        y.append(P[i,1,j])
        z.append(P[i,2,j])
        pix.append(P[i,4,j])

x = np.float32(np.array(x))
y = np.float32(np.array(y))
z = np.float32(np.array(z))
pix = np.float32(np.array(pix))
#########################################################################################33
# Saving the three vectors as csv file for executing using colab on cuml
#path = "C:/Users/Usman Khan/Dropbox/Thesis/SCITEPRESS_Conference_Latex/CSV_FILES/"
#pd.DataFrame(x).to_csv(path+"x.csv")
#pd.DataFrame(y).to_csv(path+"y.csv")
#pd.DataFrame(z).to_csv(path+"z.csv")
#pd.DataFrame(pix).to_csv(path+"pix.csv")
##########################################################################################
x_new = (x - np.min(x))/(np.max(x)-np.min(x)) * (511-0)
y_new = (y - np.min(y))/(np.max(y)-np.min(y)) * (511-0)
z_new = (z - np.min(z))/(np.max(z)-np.min(z)) * (511-0)
grand_array = np.c_[x_new,y_new,z_new,pix]
indices = np.where(grand_array[:,3] != 0)
indices = np.array(indices)
indices = indices.astype(int)
x_new = grand_array[indices,0]
y_new = grand_array[indices,1]
z_new = grand_array[indices,2]
pix_new = grand_array[indices,3]
_array = np.transpose([x_new,y_new,z_new])
array = np.concatenate((x_new,y_new,z_new),axis=0)

array = array.T
array_ = np.c_[array,pix_new.T]

##############################################################################################
# Here I need to introduce Iterations so that each of the cluster values are checked for their neighbours 
def bone_extractor(cluster):
    index = np.where(cluster[:,3]>300) # Bones have HU > 300
    index = np.array(index)
    c = cluster[index]
    return c[0,:,:]
###########################################################################################

##################################################################################################

################################################################################################
def dist_clustering(array_):
    count = 0
    cluster = np.array([])
    cluster_list = []
    empty_list = []
    while len(array_)>30:
    ##############################################################################################
        # No of clusters will be formed
        empty_list.append(array_[0].T)
        cluster = np.array(empty_list)
        x = array_[0,0:3]  # 
        x_array = np.tile(x,[array_.shape[0],1])
        ##############################################################################################################################
        diff_array = array_[:,0:3]-x_array
        
        
        ##############################################
        x_diff_array = diff_array[:,0]
        y_diff_array = diff_array[:,1]
        z_diff_array = diff_array[:,2]
        
        x_ = gpuarray.to_gpu(x_diff_array)
        y_ = gpuarray.to_gpu(y_diff_array )
        z_ = gpuarray.to_gpu(z_diff_array)
        
        
        ############################################
        out = np.empty(x_.shape,dtype="float32")
        gpu_out = gpuarray.to_gpu(out)
        var_out = gpuarray.to_gpu(out)
        
        gpuSum = ElementWiseKernel.ElementwiseKernel(
                "float *x, float *y, float *z, float *out",        
                "out[i] = x[i]*x[i] + y[i]*y[i] + z[i]*z[i]",
                
                )
        
        varKernel = ElementWiseKernel.ElementwiseKernel(
                "float *x, float y,float *out",
                "out[i] = (x[i]-y)*(x[i]-y)",
                'squared_difference'
                )
        
        gpuSum(x_,y_,z_,gpu_out)
        gpu_out = sqrt(gpu_out)
        min_val = min(gpu_out,stream = None)   
        max_val = max(gpu_out,stream = None)
        mean_val = (min_val+max_val)/2
        mean_gpu= np.float32(mean_val.get())
        varKernel(gpu_out,mean_gpu,var_out)
        n = len(var_out)
        var_out = sqrt(sum(var_out)/(n-1))
        # Here I am trying to get only those points which have a specified threshold w.r.t the mean and variance values
        # The criteria I have set is (mean-var)/2 any value lesser than this will be part of the selected point 
        # Density cluster. 
        n_idx = gpu_out<=  (mean_val.get()-var_out.get()) # Here the scale is important
        _idx = n_idx.get()
        result_ = np.where(_idx==1)
        #Npts = len(np.transpose(result_))
        
        cluster = np.vstack((cluster,array_[result_]))
    ####################################################################################################
        _idx_invert = 1 - _idx
    ###################################################################################################
    #########################################################################################################
    
    
    ######################################################################################################
        #percent_repeat = np.round(0.05*np.count_nonzero(_idx),0)
        # The idea is to repeat a random of 5% of the previous array_ in the next array_ so that it is easier to find the intersection
        # 
        count += 1
        cluster = bone_extractor(cluster)
        cluster_list.append(cluster)
        empty_list = []
    ####################################################################################################
        z = np.multiply(array_, _idx_invert[:, np.newaxis])
        array_ = z[np.all(z!=0, axis=1)]
    return cluster_list
#        z = np.multiply(array, _idx_invert[:, np.newaxis])
#        array = z[np.all(z!=0, axis=1)]
    
#######################################################################################################
# TO remove clutter from clusters we need to follow two main steps:
#    Remove all artifacts that lie only on a single plane i-e x,y or x,z or y,z and their third dimension is fixed.... we need to remove those
#    From the remaining cluster we need to accept the area which has the maximum slope in terms of color variation....    
#    Experimentation Reveals that drop the clusters with variance in x,y or z dimensions less than 20
#   We need a mask of entropy measurement in remaining clusters to segment the AOI
    

#########################################################################################################
def merge_cluster(cluster_list):
    cluster = np.vstack(cluster_list)
    return cluster
#########################################################################################################
####### Calculate variance in x, y and z dimensions to check if we have 2d patches in 3d .....
def calc_variance(cluster):
#    [var_x,var_y,var_z] = [np.var(cluster[:,0]),np.var(cluster[:,1]),np.var(cluster[:,2])]
#    return var_x,var_y,var_z
    var_z = np.var(cluster[:,2])
    return var_z
#########################################################################################################  
### Calculate the histogram
def calc_intensity_hist(cluster):
    hist,bins = np.histogram(cluster[:,3])
    return hist,bins
#########################################################################################################
def plot_hist(hist,bins):
    plt.hist(hist,bins)
    plt.xlabel('HousefieldUnits')
    plt.ylabel('Frequency')
    plt.title('cluster histogram')
    plt.xlim(0,1000)
#########################################################################################################
# Pop those arrays out of the list which have 2d patches
def pop_cluster(cluster_list):
    for i, elem in enumerate(cluster_list):
        var_z = calc_variance(cluster_list[i])
        if var_z < 2:
            cluster_list.pop(i)
#######################################################################################################

def save_pcld(filename,clutter):
    np.savetxt(filename,clutter,delimiter=',')

##########################################################################################################
def load_pcld(filename):
    return np.loadtxt(filename,delimiter=',')
############################################################################################################
def plot_mlab(cluster):
    x_ = cluster[:,0]
    y_ = cluster[:,1]
    z_ = cluster[:,2]
    pix_ = cluster[:,3]
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.points3d(x_,y_,z_,pix_, scale_mode='none', scale_factor=0.2)
    mlab.show()
##############################################################################################################
    # Function Obtained from https://devqa.io/python-compare-two-lists-of-dictionaries/
def dataframe_difference(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """
    Find rows which are different between two DataFrames.
    :param df1: first dataframe.
    :param df2: second dataframe.
    :return:    if there is different between both dataframes.
    """
    comparison_df = df1.merge(df2, indicator=True, how='outer')
    diff_df = comparison_df[comparison_df['_merge'] != 'both']
    return diff_df
###############################################################################################################
cluster_list = dist_clustering(array_)
pop_cluster(cluster_list)
cluster = merge_cluster(cluster_list)
#plot_mlab(cluster)

#clutter_list = dist_clustering(cluster)
#pop_cluster(clutter_list)
#clutter = merge_cluster(clutter_list)
##############################################################################################################





#mlab.show()
##########################################################################################################################3###
def mutual_info(array_):
    cl = dist_clustering(array_)
    cl = np.array(cl)
    b = len(cl[0])
    for i in range(len(cl)):
        if b < len(cl[i]):
            b = len(cl[i])
    c_array = np.zeros([len(cl),b])
    A = np.zeros([len(cl),len(cl)])
    index = np.zeros(len(cl))
    for i in range(len(cl)):
        zero_padded = np.zeros(b)
        if len(cl[i]) < b:
            idx = np.where(cl[i][:,3])
            zero_padded[idx] = cl[i][:,3]
            c_array[i,idx] = zero_padded[idx]
    for i in range(len(cl)):
        for j in range(len(cl)):
            if i != j:
                A[i,j] = nmis(c_array[i,:],c_array[j,:])
    index = np.where(A>0.2)
    z = np.zeros(A.shape)
    z[index] = A[index]
    return(A,cl,index,z)
###############################################################################################################################    
def show_NMIS_MAP(z):
    plt.imshow(z)
    plt.xlabel("clusters---->>")
    plt.ylabel("clusters---->>")
    plt.title("NMIS MAP")

##############################################################################################################################

def get_nmis_clusters(index,cl):
    idx = np.array(index)
    idx = idx.T
    ix = np.unique(idx[:,0])
    new_clusters = list()
    for k in range(len(ix)):
        cc = idx[np.where(idx[:,0] == ix[k]),1]    
        cc = idx[np.where(idx[:,0] == ix[k]),1]
        ii = np.append(cc,ix[k])  
        sub_cl = cl[ii]
        s_c = merge_cluster(sub_cl)
        new_clusters.append(s_c)
    return (new_clusters)
########################################################################################
[A,cl,index,z] = mutual_info(cluster)
new_clusters = get_nmis_clusters(index,cl)
for i in range(len(new_clusters)):
    cluster_list = cluster.tolist()
    clutter_list = new_clusters[i]
    
    df_cluster = pd.DataFrame(cluster_list)
    df_clutter = pd.DataFrame(clutter_list)
    
    diff = dataframe_difference(df_cluster, df_clutter)
    
    cluster = diff.values
    cluster = cluster[:,0:4]
    cluster = cluster.astype(np.float32)
plot_mlab(cluster)

###################################################################################################################
### score = sklearn.feature_selection.mutual_info_classif(X, y, *, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
### while (c is not empty)
    ### if score > threshold :
        #   merge_cluster_(c,c1)
    ### else:
        #   merge_cluster_(clutter,c1)


##############################################################################################################################

#mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
#pts = mlab.points3d(x_new, y_new, z_new, pix_new, mode='point', colormap = "CMRmap", scale_factor=0.2)
#mlab.show()
################################################################################################################################
#mesh = mlab.pipeline.delaunay2d(pts)
#surf = mlab.pipeline.contour_surface(mesh)
#mlab.show()
#scatter = dict(
#    mode = "markers",
#    name = "y",
#    type = "scatter3d",
#    x = x_new, y = y_new, z = z_new,
#    marker = dict( size=2, color="rgb(23, 190, 207)" )
#)
#clusters = dict(
#    alphahull = 7,
#    name = "y",
#    opacity = 0.1,
#    type = "mesh3d",
#    x = x_new, y = y_new, z = z_new
#)
#layout = dict(
#    title = '3d point clustering',
#    scene = dict(
#        xaxis = dict( zeroline=False ),
#        yaxis = dict( zeroline=False ),
#        zaxis = dict( zeroline=False ),
#    )
#)
#fig = dict( data=[scatter, clusters], layout=layout )
## Use py.iplot() for IPython notebook
#py.iplot(fig, filename='3d point clustering')
###############################################################################################################################

