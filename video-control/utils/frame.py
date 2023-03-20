import cv2
import numpy as np

class FrameGenerator:

    def __init__(self, device : int) -> None:
        self.device = device
        self.running = False
        
    def __call__(self,):
        cap = cv2.VideoCapture(self.device)
        self.running = True
# def gen_frames(cap):

        while self.running:
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')