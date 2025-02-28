import traceback

try:
    import predict
    print("Import successful")
except Exception as e:
    traceback.print_exc()
    print(f"Error: {str(e)}")
