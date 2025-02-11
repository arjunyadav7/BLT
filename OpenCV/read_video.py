import cv2 as cv
video_capture = cv.VideoCapture('D:/Archisha/Seed.1994.720p.BRRip.Hindi.Dual-Audio.Vegamovies.NL.mkv')

res = True
while res:
    res, frame = video_capture.read()

    if res:
        cv.imshow('video frame', frame)
        cv.waitKey(4)
    
video_capture.release()
cv.destroyAllWindows()    
