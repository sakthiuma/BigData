import os
import pickle
import sys

import cv2 as cv
import pymongo
from bson.binary import Binary


def image_similarity(refImageName, imageToCheckName, feature_collection):
    rootPath = "C:/Users/sakth/OneDrive/Documents/NYU courses/Big Data/Project/dataset/"
    refImage = cv.imread(os.path.join(rootPath, refImageName))
    imageToCheckSimilarity = cv.imread(os.path.join(rootPath, imageToCheckName))

    refImage = cv.cvtColor(refImage, cv.COLOR_BGR2GRAY)
    imageToCheckSimilarity = cv.cvtColor(imageToCheckSimilarity, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()

    # Retrieving the image description from the db if already present to avoid recomputation
    refImageDes = feature_collection.find_one({"name": refImageName}, {"description": 1})
    if refImageDes is None:
        keypoint1, refImageDes = sift.detectAndCompute(refImage, None)
        keypoint_json = [{'angle': k.angle, 'response': k.response} for k in keypoint1]
        img_dict = {"name": refImageName, "keypoint": keypoint_json, "description": Binary(pickle.dumps(refImageDes, protocol=2), subtype=128)}
        feature_collection.insert_one(img_dict)
    else:
        refImageDes = pickle.loads(refImageDes["description"])

    imageToCheckDes = feature_collection.find_one({"name" : imageToCheckName}, {"description" : 1})
    if imageToCheckDes is None:
        keypoint, imageToCheckDes = sift.detectAndCompute(imageToCheckSimilarity, None)
        keypoint_json = [{'angle': k.angle, 'response': k.response} for k in keypoint]
        img_dict = {"name": imageToCheckName, "keypoint": keypoint_json, "description": Binary(pickle.dumps(imageToCheckDes, protocol=2), subtype=128)}
        feature_collection.insert_one(img_dict)
    else:
        imageToCheckDes = pickle.loads(imageToCheckDes["description"])


    # compute the image hash and store it in a separate db.
    # if the image hash is already present use it else call detect and compute for that image
    # this is to avoid calling the image description again for the same corpus for performing
    # similarity calculation for other images in the corpus
    # so the first time we compute for image1, and the next time we want to compute for image2
    # we would avoid computation of desc and use the value that is already stored.

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(refImageDes, imageToCheckDes)

    matches = sorted(matches, key=lambda x: x.distance)
    sum_matches = sum([i.distance for i in matches])
    minMatch = min([i.distance for i in matches])
    return (((127-len(matches)) * minMatch) + sum_matches)/127


if __name__ == "__main__":
    refImagePath = sys.argv[1] if len(sys.argv) > 1 else "images_4.jpeg"
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client["image_features_db"]
    Collection = db["features"]
    for similarImgPath in sys.stdin:
        similarImgPath = similarImgPath.strip()
        similarityScore = image_similarity(refImagePath.strip(), similarImgPath, Collection)
        print('%d%s%s%s%s' % (similarityScore, '\t', similarImgPath.strip(), '\t', refImagePath))

# assumption the images names should be unique
# have to check if image size has any limitations