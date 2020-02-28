import numpy as np
import cv2
def PCA(data, correlation = False, sort = True):
    """ Applies Principal Component Analysis to the data

    Parameters
    ----------        
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix. 

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.   
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors

def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------        
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])

    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.    
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]

    #: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal

def demo():
    p,n =best_fitting_plane(np.array([[0,0,0],[1,0,0],[0,0,1]]))
    print(p)
    print(n)
def getCameraPose(src):
    f = open(src)
    lines = f.readlines()
    mvp = np.zeros((4,4))
    i =  0
    for line in lines[:-1]: 
        mvp[i] = np.array(line[:-1].split(' ')).astype(np.float) 
        i = i+1
    cc = np.array(lines[-1].split(' ')).astype(np.float)
    return mvp, cc
def drawPoints(blended, con, gray):
    p_num = int(len(con[0]) / 3)
    for i in range(p_num):
        index = con[0][3*i]
        y = int(index / 640)
        x = index - y * 640
        # print(str(y) + " " + str(x))
        cv2.circle(blended, (x,y), 5, (0,255,0), 2)

def main(mask, cloud, blended, mvp):
    pids = np.unique(mask)[1:]
    pns = []
    
    for pid in pids:
        sc = cloud[np.where(mask == pid)]
        con = np.where(sc!=[.0,.0,.0])
        ppoints = sc[con]

        p_num = len(ppoints) / 3
        if(p_num < 3):
            continue
        #draw points
        # drawPoints(blended, con, pid)

        p, n = best_fitting_plane(ppoints.reshape(int(p_num), 3))
        print("plane " + str(pid))
        print(p)
        p = np.append(p, 1.0)
        proj = np.dot(mvp, p)
        proj = proj / proj[-1]
        print(proj)
        if(proj[0] < -1.0 or proj[0]>1.0 or proj[1]<-1.0 or proj[1]>1.0):
            print("out bound")
        else:
            y = int ((1.0 - (proj[1] * 0.5 + 0.5)) * 480)
            x = int ((proj[0] * 0.5 + 0.5) * 640)
            cv2.circle(blended, (x,y), 20, (255, 0, 0), 2)
            print("circle at " + str(y) +" "+str(x))
            
        print(n)
        pns.append(n)
    # for i in range(len(pns)):
    #     for j in range(i+1, len(pns)):
    #         cos = np.dot(pns[i], pns[j])
    #         print(cos)
    cv2.imwrite("test.png", blended)

if __name__ == '__main__':
    # src = "../../mediapipe/mappoints/0221/91.txt"
    # mask = "../../PEAC/plane_seg/328.png"
    cloud = np.load("../res/0221/70.npz.npy")
    mask = cv2.imread("../../PEAC/plane_seg/70.png", cv2.IMREAD_GRAYSCALE)
    mvp, cc = getCameraPose("../../mediapipe/camera/0221/70.txt")
    blended = cv2.imread("_blend/70.png")
    main(mask, cloud, blended, mvp)