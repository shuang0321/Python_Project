'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = '/Users/huangshuhui/Desktop/study/CS289/hw2017/hw1/hw01_data/spam/'
# SPAM_DIR = 'spam/'
SPAM_DIR = 'spam/'
# HAM_DIR = 'ham/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def freq_lott_feature(text, freq):
    return text.count('lott' )

def freq_awarded_feature(text, freq):
    return text.count('awarded' )   

def freq_unsubscribe_feature(text, freq):
    return text.count('unsubscribe' ) 

def freq_medication_feature(text, freq):
    return text.count('medication' ) 

def freq_unsub_feature(text, freq):
    return text.count('unsub' ) 

def freq_free_feature(text, freq):
    return text.count('free')

def freq_lower_feature(text, freq):
    return text.count('lower')

def freq_tired_feature(text, freq):
    return text.count('tired')

def freq_payless_feature(text, freq):
    return text.count('pay less')

def freq_limited_feature(text, freq):
    return text.count('limited')

def freq_sex_feature(text, freq):
    return text.count('sex')

def freq_cheap_feature(text, freq):
    return text.count('cheap')

def freq_save_feature(text, freq):
    return text.count('save')

def freq_amazing_feature(text, freq):
    return text.count('amazing')

def freq_brand_feature(text, freq):
    return text.count('brand')

def freq_products_feature(text, freq):
    return text.count('products')

def freq_ask_feature(text, freq):
    return text.count('?')

def freq_cialis_feature(text, freq):
    return text.count('cialis')

def freq_cost_feature(text, freq):
    return text.count('cost')

def freq_offering_feature(text, freq):
    return text.count('offering')

def freq_moneyback_feature(text, freq):
    return text.count('money back')

def freq_available_feature(text, freq):
    return text.count('available')

def freq_penis_feature(text, freq):
    return text.count('penis')

def freq_click_feature(text, freq):
    return text.count('click')

def freq_off_feature(text, freq):
    return text.count('off')

def freq_investment_feature(text, freq):
    return text.count('investment')

def freq_card_feature(text, freq):
    return text.count('card')

def freq_viagra_feature(text, freq):
    return text.count('viagra')

def freq_mortgage_feature(text, freq):
    return text.count('mortgage')

def freq_offer_feature(text, freq):
    return text.count('offer')

def freq_discount_feature(text, freq):
    return text.count('discount')

def freq_quick_feature(text, freq):
    return text.count('quick')

def freq_risk_feature(text, freq):
    return text.count('risk')

def freq_online_feature(text, freq):
    return text.count('online')

def freq_million_feature(text, freq):
    return text.count('million')

def freq_pay_feature(text, freq):
    return text.count('pay')

def freq_deliver_feature(text, freq):
    return text.count('deliver')

def freq_price_feature(text, freq):
    return text.count('price')

def freq_remove_feature(text, freq):
    return text.count('remove')

def freq_guaranteed_feature(text, freq):
    return text.count('guaranteed')
    
def freq_pill_feature(text, freq):
    return text.count('pill')

def freq_order_feature(text, freq):
    return text.count('order')

def freq_urgent_feature(text, freq):
    return text.count('urgent')

def freq_only_feature(text, freq):
    return text.count('only')

def freq_deal_feature(text, freq):
    return text.count('deal')

def freq_transfer_feature(text, freq):
    return text.count('transfer')

def freq_approved_feature(text, freq):
    return text.count('approved')

def freq_adult_feature(text, freq):
    return text.count('adult')
    
def freq_http_feature(text, freq):
    return text.count('http')

def freq_discover_feature(text, freq):
    return text.count('discover')
    
def freq_best_feature(text, freq):
    return text.count('best')

def freq_ad_feature(text, freq):
    return text.count('ad')

def freq_special_feature(text, freq):
    return text.count('special')

def freq_fuck_feature(text, freq):
    return text.count('fuck')

def freq_paliourg_feature(text, freq):
    return text.count('paliourg')

def freq_dollar2_feature(text, freq):
    return text.count('dollar')

def freq_oem_feature(text, freq):
    return text.count('oem')

def freq_weight_feature(text, freq):
    return text.count('weight')
    
def freq_stock_feature(text, freq):
    return text.count('stock')

def freq_hesitate_feature(text, freq):
    return text.count('hesitate')

def freq_software_feature(text, freq):
    return text.count('software')
    
def freq_latest_feature(text, freq):
    return text.count('latest')

def freq_premium_feature(text, freq):
    return text.count('premium')

def freq_hot_feature(text, freq):
    return text.count('hot')

def freq_prefer_feature(text, freq):
    return text.count('prefer')

def freq_overnight_feature(text, freq):
    return text.count('overnight')
    
def freq_stop_feature(text, freq):
    return text.count('stop')

def freq_loan_feature(text, freq):
    return text.count('loan')

def freq_discreet_feature(text, freq):
    return text.count('discreet')

def freq_body_feature(text, freq):
    return text.count('body')

def freq_pic_feature(text, freq):
    return text.count('pic')

def freq_woman_feature(text, freq):
    return text.count('woman')
    
def freq_important_feature(text, freq):
    return text.count('important')

def freq_update_feature(text, freq):
    return text.count('update')

def freq_profile_feature(text, freq):
    return text.count('profile')

def freq_med_feature(text, freq):
    return text.count('med')

def freq_guranteed_feature(text, freq):
    return text.count('guranteed')
    
def freq_embarrassment_feature(text, freq):
    return text.count('embarrassment')

def freq_confirm_feature(text, freq):
    return text.count('confirm')

def freq_fast_feature(text, freq):
    return text.count('fast')

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    feature.append(freq_lott_feature(text, freq))
    feature.append(freq_awarded_feature(text, freq))
    feature.append(freq_unsubscribe_feature(text, freq))
    feature.append(freq_medication_feature(text, freq))
    feature.append(freq_unsub_feature(text, freq))
    feature.append(freq_free_feature(text, freq))
    feature.append(freq_lower_feature(text, freq))
    feature.append(freq_tired_feature(text, freq))
    feature.append(freq_payless_feature(text, freq))
    feature.append(freq_limited_feature(text, freq))
    feature.append(freq_sex_feature(text, freq))
    feature.append(freq_cheap_feature(text, freq))
    feature.append(freq_save_feature(text, freq))
    feature.append(freq_amazing_feature(text, freq))
    feature.append(freq_brand_feature(text, freq))
    feature.append(freq_products_feature(text, freq))
    feature.append(freq_ask_feature(text, freq))
    feature.append(freq_cialis_feature(text, freq))
    feature.append(freq_cost_feature(text, freq))
    feature.append(freq_offering_feature(text, freq))
    feature.append(freq_moneyback_feature(text, freq))
    feature.append(freq_available_feature(text, freq))
    feature.append(freq_penis_feature(text, freq))
    feature.append(freq_click_feature(text, freq))
    feature.append(freq_off_feature(text, freq))
    feature.append(freq_investment_feature(text, freq))
    feature.append(freq_card_feature(text, freq))
    feature.append(freq_viagra_feature(text, freq))
    feature.append(freq_mortgage_feature(text, freq))
    feature.append(freq_offer_feature(text, freq))
    feature.append(freq_discount_feature(text, freq))
    feature.append(freq_quick_feature(text, freq))
    feature.append(freq_risk_feature(text, freq))
    feature.append(freq_online_feature(text, freq))   
    feature.append(freq_million_feature(text, freq))
    feature.append(freq_pay_feature(text, freq))
    feature.append(freq_deliver_feature(text, freq))
    feature.append(freq_price_feature(text, freq))
    feature.append(freq_remove_feature(text, freq))
    feature.append(freq_guaranteed_feature(text, freq))
    feature.append(freq_pill_feature(text, freq))
    feature.append(freq_order_feature(text, freq))   
    feature.append(freq_urgent_feature(text, freq))
    feature.append(freq_only_feature(text, freq))
    feature.append(freq_deal_feature(text, freq))
    feature.append(freq_transfer_feature(text, freq))
    feature.append(freq_approved_feature(text, freq))
    feature.append(freq_adult_feature(text, freq))
    feature.append(freq_http_feature(text, freq))
    feature.append(freq_discover_feature(text, freq))
    feature.append(freq_best_feature(text, freq))
    feature.append(freq_ad_feature(text, freq))
    feature.append(freq_special_feature(text, freq))
    feature.append(freq_fuck_feature(text, freq))
    feature.append(freq_paliourg_feature(text, freq))   
    feature.append(freq_dollar2_feature(text, freq))
    feature.append(freq_oem_feature(text, freq))
    feature.append(freq_weight_feature(text, freq))
    feature.append(freq_stock_feature(text, freq))
    feature.append(freq_hesitate_feature(text, freq))
    feature.append(freq_software_feature(text, freq))
    feature.append(freq_latest_feature(text, freq))
    feature.append(freq_premium_feature(text, freq))
    feature.append(freq_hot_feature(text, freq))
    feature.append(freq_prefer_feature(text, freq))
    feature.append(freq_overnight_feature(text, freq))
    feature.append(freq_stop_feature(text, freq))
    feature.append(freq_loan_feature(text, freq))
    feature.append(freq_discreet_feature(text, freq))
    feature.append(freq_body_feature(text, freq))
    feature.append(freq_pic_feature(text, freq))
    feature.append(freq_woman_feature(text, freq))
    feature.append(freq_important_feature(text, freq))
    feature.append(freq_update_feature(text, freq))
    feature.append(freq_profile_feature(text, freq))
    feature.append(freq_med_feature(text, freq))
    feature.append(freq_guranteed_feature(text, freq))
    feature.append(freq_embarrassment_feature(text, freq))
    feature.append(freq_confirm_feature(text, freq))
    feature.append(freq_fast_feature(text, freq))   
 
    # Make sure type is int or float

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
            with open(filename) as f:
                try:
                    text = f.read() # Read in text from file
                    text = text.replace('\r\n', ' ') # Remove newline character
                    words = re.findall(r'\w+', text)
                    word_freq = defaultdict(int) # Frequency of all wordz
                except:
                    pass
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data1.mat', file_dict)

