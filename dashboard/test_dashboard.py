"""Focused checks: frozen forecasts, metric parity, availability cutoff, API scope."""
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from server import Data
class DashboardTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.data=Data()
 def test_final_forecast_and_no_spot_in_response(self):
  for period in ['validation','test']:
   o=pd.Timestamp(self.data.origins(period)[100]);r=self.data.forecast(period,o)
   self.assertEqual(len(r['rows']),40)
   original=self.data.frames[period];original=original[original.forecast_origin_utc==o].sort_values('step')
   self.assertEqual([x['p50'] for x in r['rows']],original.p50.tolist())
   self.assertNotIn('spot_price',r['rows'][0])
   self.assertIsNone(self.data.forecast(period,pd.Timestamp(self.data.origins(period)[0]))['rows'][0]['previous_p50'])
 def test_full_period_reproduces_saved_scores(self):
  for period,mae,pinball,rows in [('test',26.702820052005947,6.830205926263002,26920),('validation',41.59882833333333,11.886367109363295,26700)]:
   r=self.data.performance(period,pd.Timestamp(self.data.origins(period)[0]),0,'full')
   self.assertEqual(r['rows'],rows)
   self.assertAlmostEqual(r['final']['mae_p50'],mae,places=8)
   self.assertAlmostEqual(r['final']['calibration_pinball_mean'],pinball,places=8)
 def test_asof_metrics_ignore_future_outcomes(self):
  p='test';o=pd.Timestamp(self.data.origins(p)[200]);before=self.data.performance(p,o,7,'asof')
  f=self.data.joined[p];future=f.usable>o;saved=f.loc[future,'imbalance_price'].copy()
  try:
   f.loc[future,'imbalance_price']=999999
   after=self.data.performance(p,o,7,'asof')
   self.assertEqual(before,after)
  finally:f.loc[future,'imbalance_price']=saved
 def test_fresh_checkout_reconstructs_test_actuals(self):
  original=Path.exists
  def exists(path):
   if path.name=='local_actuals_test.csv':return False
   return original(path)
  with patch.object(Path,'exists',exists):
   fresh=Data()
  result=fresh.performance('test',pd.Timestamp(fresh.origins('test')[0]),0,'full')
  self.assertEqual(result['rows'],26920)
  self.assertAlmostEqual(result['final']['mae_p50'],26.702820052005947,places=8)
 def test_first_origin_has_no_historical_scores(self):
  for p in ['validation','test']:
   r=self.data.performance(p,pd.Timestamp(self.data.origins(p)[0]),7,'asof')
   self.assertIsNone(r['final']);self.assertEqual(r['rows'],0)
if __name__=='__main__':unittest.main()
