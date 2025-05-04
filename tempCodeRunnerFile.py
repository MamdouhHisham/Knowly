import cv2
from core.utils import get_video_feed
from face_analyzer import FaceAnalyzer

def main():
    analyzer = FaceAnalyzer()
    cap = get_video_feed()
    forward_center=0
    frame_count = 0
    focus=False
    if not cap.isOpened():
        print("Error: Could not open video feed")
        return
    
    analyzer.start_session()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video feed ended")
                break
                
            results, annotated_frame,focus = analyzer.analyze(frame)
            if focus==True:
                forward_center+=1
            display_frame = annotated_frame.copy()
            frame_count+=1
            # Display text on screen
            y_pos = 30
            if results:
                if results["head_pose"]:
                    hp = results["head_pose"]
                    head_text = f"Head: {hp['orientation'].upper()} (Yaw={hp['yaw']:.1f}, Pitch={hp['pitch']:.1f})"
                    cv2.putText(display_frame, head_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25

                if results["gaze"]["horizontal"] is not None:
                    gaze = results["gaze"]
                    direction = "Left" if gaze["is_left"] else "Right" if gaze["is_right"] else "Center"
                    gaze_text = f"Gaze: {direction} (H:{gaze['horizontal']:.2f}, V:{gaze['vertical']:.2f})"
                    if gaze["is_blinking"]:
                        gaze_text += " [Blink]"
                    cv2.putText(display_frame, gaze_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25

                if results["emotion"]:
                    cv2.putText(display_frame, f"Emotion: {results['emotion'].upper()}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Child Attention Analysis", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if analyzer.save_report():
            print("\nSession report saved successfully")
            report = analyzer.generate_report()
            print(f"Session Duration: {report['session_info']['duration_seconds']:.2f} seconds")
            try:
                print("focused",(forward_center*100)/frame_count,"%")
            except:ZeroDivisionError
        else:
            print("\nFailed to save session report")

if __name__ == "__main__":
    main()