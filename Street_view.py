import google_streetview.api
import pandas as pd
import os


df = pd.read_excel('x1.xlsx')
df = df.astype(str)

coords = ';'.join([r.POINT_Y + ',' + r.POINT_X for _, r in df.iterrows()])

params = {
	'size': '640x300',
	'location': coords,
	'heading': '0;90;180;270',
	'key': 'secret'
}

api_list = google_streetview.helpers.api_list(params)
results = google_streetview.api.results(api_list)
results.download_links('downloads')
results.save_metadata('metadata.json')

