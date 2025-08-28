from src.pipline.training_pipeline import TrainPipeline

pipeline=TrainPipeline()
pipeline.run_pipeline()

# Testing of prediction pipeline

# from src.pipline.prediction_pipeline import SpotifyData,PredictionPipeline
   
# test_song = SpotifyData(
#     danceability=0.82,      # very danceable
#     energy=0.85,            # high energy
#     key=5,                  # F major (common key in pop)
#     loudness=-4.5,          # loud, professionally mixed
#     mode=1,                 # major (happier sound)
#     speechiness=0.06,       # low speech (not rap)
#     acousticness=0.08,      # very low acoustic, more produced
#     instrumentalness=0.0,   # almost no instrumental, vocals-driven
#     liveness=0.12,          # some live feel but not too much
#     valence=0.75,           # positive, happy vibe
#     tempo=124.0,            # common EDM/pop tempo
#     duration_ms=205000,     # ~3 min 25 sec (radio-friendly length)
#     time_signature=4,       # standard
#     chorus_hit=42.0,        # early chorus (good for catchiness)
#     sections=11             # enough structure/variation
# )



# # # print(spotify_sample.__dict__)
# # # df=spotify_sample.get_data_as_dataframe()
# # # print(df.head())
# prediction_pipeline=PredictionPipeline()
# status=prediction_pipeline.predict(test_song)
# print(status)
