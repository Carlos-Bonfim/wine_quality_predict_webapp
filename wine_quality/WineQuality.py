import pickle
import numpy as np

class WineQuality( object ):
	def __init__ ( self ):
		self.free_sulfur_scaler = pickle.load( open( 'parameter/free_sulfur_scaler.pkl', 'rb' ) )
		self.total_sulfur_scaler = pickle.load( open( 'parameter/total_sulfur_scaler.pkl', 'rb' ) )


	def data_preparation( self, df ):
		# rescalling free sulfur
		df['free sulfur dioxide'] = self.free_sulfur_scaler.transform( df[['free sulfur dioxide']].values )

		# rescalling total sulfur
		df['total sulfur dioxide'] = self.total_sulfur_scaler.transform( df[['total sulfur dioxide']].values )

		return df