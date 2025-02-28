import traceback

try:
    from predict import create_gradio_interface
    print("Import successful")
    
    try:
        app = create_gradio_interface()
        print("Interface created")
        
        try:
            print("Attempting to launch interface...")
            app.launch(share=False, inbrowser=False)
            print("Interface launched successfully")
        except Exception as e:
            print(f"Error launching interface: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error creating interface: {str(e)}")
        traceback.print_exc()
except Exception as e:
    print(f"Error importing: {str(e)}")
    traceback.print_exc()
