import boto3
from autocorrect import spell

if __name__ == "__main__":

    bucket='vb7mzdrsinus2'
    photo='ugly.jpg'

    client=boto3.client('rekognition')

  
    response=client.detect_text(Image={'S3Object':{'Bucket':bucket,'Name':photo}})

                        
    textDetections=response['TextDetections']
    #print(response)
    print('Matching faces')
    for text in textDetections:
            print(text['DetectedText']+" ---- "+spell(text['DetectedText']))
            #print('Confidence: ' + "{:.2f}".format(text['Confidence']) + "%")
            #print('Id: {}'.format(text['Id']))
            #if 'ParentId' in text:
            #    print('Parent Id: {}'.format(text['ParentId']))
            #print('Type:' + text['Type'])