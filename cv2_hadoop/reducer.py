import sys
import pymongo


def getImageName(line):
    return line.split('\t')[1]


def getSimilarityScore(line):
    return line.split('\t')[0]


def getRefImage(line):
    return line.split('\t')[2]


def main(separator='\t'):
    kCount = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    innerThreshold = 1000
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client["similar_images_db"]
    similar_images_collection = db["similar_images"]
    images_list = None
    refImageName = ""
    priorLen = -1
    for line in sys.stdin:
        if images_list is None:
            refImageName = getRefImage(line)
            similar_image_list = similar_images_collection.find_one({"refImage": getRefImage(line).strip()}, {"similar_images":1, "_id" : 0})
            print("ref image" , getRefImage(line))
            if similar_image_list is None:
                images_list = []
            else:
                images_list = similar_image_list["similar_images"]
                #print(" image list already present" , type(images_list), len(images_list))
            priorLen = len(images_list)

        if innerThreshold > 0:
            dictVal = {"image": getImageName(line), "score": getSimilarityScore(line)}
            if dictVal not in images_list:
                images_list.append(dictVal)
                innerThreshold = innerThreshold - 1

            if kCount > 0:
                print('%s' % (line))
                kCount = kCount-1

    print(priorLen,  len(images_list))
    if priorLen != len(images_list):
        input = {"refImage" : refImageName.strip(), "similar_images": images_list}
        similar_images_collection.insert_one(input)

    # here we have to store the similarity for not just the n images that the use wants but for
    # a threshold value that we have decided which provides the user with the flexibity to see the
    # next set of n images needed.
    # i.e this becomes handy when we want to fetch the next set of unseen similar images for the
    # model to train. This is an extra optimization that we are performing.


if __name__ == "__main__":
    main()

# hadoop jar C:/hadoop-3.2.2/hadoop-streaming-2.7.3.jar
# -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator
# -D  mapred.text.key.comparator.options=-n
# -input input.txt -output ./output1
# -mapper "python mapper.py road121.png" -reducer "python reducer.py"
