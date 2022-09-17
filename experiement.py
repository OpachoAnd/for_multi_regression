# from hackaton import Zinc_Impurities
from prepare_df import Prepare_Df
from prepare_df_test import Prepare_Df_Test
from normalization_df import Normalization_Df
import redis

path_train = r'C:\Users\Андрей\Documents\GitHub\for_multi_regression\source_data\out.csv'
path_test = r'C:\Users\Андрей\Documents\GitHub\for_multi_regression\source_data\check_input.xls'
path_out = r'C:\Users\Андрей\Documents\GitHub\for_multi_regression\source_data\time_point.xls'
#
# prepare_df = Prepare_Df(path=path_test,
#                         name_target_column_cuprum='Cu_AT502',
#                         name_target_column_cadmium='Cd_AT502')
#
# normalization_df = Normalization_Df(df=prepare_df.df, test=True)
# normalization_df.normalization()
# print(normalization_df.offset)

prepare_df = Prepare_Df_Test(path_test_df=path_test,
                             path_time_point=path_out,
                             name_target_column_cuprum='Cu_AT502',
                             name_target_column_cadmium='Cd_AT502')
print(prepare_df.selecting_df)
