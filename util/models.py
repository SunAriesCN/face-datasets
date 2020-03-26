import os

try:
    from caffe_extractor import CaffeExtractor
except:
    print("It's not in Caffe Environment.")
    CaffeExtractor = None

if CaffeExtractor is None:
    try:
        from tflite_extractor import TFLiteExtractor
    except:
        print("There is no suitable environment.")
        
from mobileid.mobile_id import MobileID
from mobileface.mobilefacenet import MobileFaceNet
        
def model_centerface(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'centerface')
    model_proto = model_dir + 'face_deploy.prototxt'
    model_path = model_dir + 'face_model.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size
    
def model_sphereface(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'sphereface')
    model_proto = model_dir + 'sphereface_deploy.prototxt'
    model_path = model_dir + 'sphereface_model.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size
    
def model_AMSoftmax(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'AMSoftmax')
    if do_mirror:
        model_proto = model_dir + 'face_deploy_mirror_normalize.prototxt'
    else:
        model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'face_train_test_iter_30000.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = False, featLayer='fc5')
    return extractor, image_size
    
    
def model_arcface(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'arcface')
    model_proto = model_dir + 'model.prototxt'
    model_path = model_dir + 'model-r50-am.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size
    

def model_mobileface(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'mobilefacenet')
    model_proto = os.path.join(model_dir, 'mobilefacenet-res2-6-10-2-dim128-opencv.prototxt')
    model_path = os.path.join(model_dir, 'mobilefacenet-res2-6-10-2-dim128.caffemodel')
    image_size = (112, 112)
#     extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    extractor = MobileFaceNet(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size

def model_lattemindface(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'lattemindface')
    model_path = os.path.join(model_dir, 'optimized_graph.tflite')
    image_size = (112, 112)
    extractor = TFLiteExtractor(model_path)
    return extractor, image_size
        
def model_mobileid(model_dir, do_mirror):
    model_dir = os.path.join(model_dir, 'mobileid')
    model_proto = os.path.join(model_dir, 'mobile_id_gallery.prototxt')
    model_path = os.path.join(model_dir, 'mobile_id.caffemodel')
    image_size = (112,112)
    extractor = MobileID(model_proto, model_path, do_mirror = do_mirror, featLayer='ip3')
    return extractor, image_size
    
def model_yours(model_dir, do_mirror):
    model_dir = '/path/to/your/model/'
    model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'weights.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size

def model_factory(name, model_dir, do_mirror=False):
    model_dict = {
        'centerface':model_centerface, 
        'sphereface':model_sphereface, 
        'AMSoftmax' :model_AMSoftmax, 
        'arcface'   :model_arcface,
        'mobileface':model_mobileface, 
        'lattemindface':model_lattemindface,
        'mobileid':model_mobileid,
        'yours'     :model_yours, 
    }
    model_func = model_dict[name]
    return model_func(model_dir, do_mirror) 
