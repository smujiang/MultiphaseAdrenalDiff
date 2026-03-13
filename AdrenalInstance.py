import glob
import os
from numpy import mean
import pandas as pd
from requests import post

from utils import *

PHASE_ORDER = ["NC", "AR", "PV", "Delay"]

class AdrenalInstance:
    def __init__(self, MRN, StudyDate, HU_data_dir, Image_data_dir): 
        self.MRN = MRN
        self.StudyDate = StudyDate
        self.HU_data_dir = HU_data_dir
        self.Image_data_dir = Image_data_dir

        self.all_available_phases = []
        self.phase_delta_t = {}
        self.phase_HU_arr_files = {}
        self.phase_mask_arr_files = {}
        self.phase_img_files = {}
        self.Instance_init()

    def Instance_init(self):
        mask_arr_files = glob.glob(os.path.join(self.HU_data_dir, self.MRN + "_" + self.StudyDate + "*_mask.npy"))
        HU_arr_files = [f.replace("_mask.npy", ".npy") for f in mask_arr_files]
        img_files = [os.path.join(self.Image_data_dir, os.path.basename(f).replace(".npy", ".jpg")) for f in HU_arr_files] 
        for idx, HU_arr_fn in enumerate(HU_arr_files):
            phase, t = self._get_phase_info_from_fn(HU_arr_fn)
            self.phase_HU_arr_files[phase] = HU_arr_fn
            self.phase_mask_arr_files[phase] = HU_arr_fn.replace(".npy", "_mask.npy")
            self.phase_img_files[phase] = img_files[idx]
            self.phase_delta_t[phase] = t
            self.all_available_phases.append(phase)
        self.all_available_phases.sort(key=lambda x: PHASE_ORDER.index(x))
        # in case there are two delay phases, only keep one
        Delay_count = self.all_available_phases.count("Delay")
        if Delay_count > 1:
            print(f"Warning: Multiple 'Delay' phases found for MRN {self.MRN}, StudyDate {self.StudyDate}. ")
            
    def is_washout_available(self):
        if "NC" in self.all_available_phases and "PV" in self.all_available_phases and "Delay" in self.all_available_phases:
            return True
        return False
    
    def _get_phase_info_from_fn(self, HU_arr_fn):
        base_fn = os.path.basename(HU_arr_fn)
        ele = base_fn.split('_')
        phase = ele[2]
        t = ele[3]
        return phase, float(t)
    
    def calculate_Average_HU(self):
        avg_HU = {}
        for phase in self.all_available_phases:
            HU_img = self.phase_HU_arr_files[phase]
            lesion_mask_img = self.phase_mask_arr_files[phase]
            avg_HU[phase] = calculate_lesion_HU(HU_img, lesion_mask_img)
        return avg_HU
    
    def get_attenuation_features(self, return_as_dict=True):
        if not self.is_washout_available():
            return [None,None,None,None]
        else:
            pre_HU_img = self.phase_HU_arr_files["NC"]
            post_HU_img = self.phase_HU_arr_files["PV"]
            delayed_HU_img = self.phase_HU_arr_files["Delay"]
            lesion_mask_img = self.phase_mask_arr_files["NC"]
            delay_time = self.phase_delta_t["Delay"]
            absolute_washout = calculate_absolute_washout(pre_HU_img, post_HU_img, delayed_HU_img, lesion_mask_img)
            relative_washout = calculate_relative_washout(post_HU_img, delayed_HU_img, lesion_mask_img)
            absolute_washout_rate = calculate_absolute_washout_rate(pre_HU_img, post_HU_img, delayed_HU_img, lesion_mask_img, delay_time)
            relative_washout_rate = calculate_relative_washout_rate(post_HU_img, delayed_HU_img, lesion_mask_img, delay_time)
            if return_as_dict:
                return {
                    "absolute_washout": absolute_washout,
                    "relative_washout": relative_washout,
                    "absolute_washout_rate": absolute_washout_rate,
                    "relative_washout_rate": relative_washout_rate
                }
            else:
                return [absolute_washout, relative_washout, absolute_washout_rate, relative_washout_rate]

    def get_morphological_features(self, phase, return_as_dict=True):
        if phase not in self.all_available_phases:
            return [None, None, None, None, None]
        mask_arr_file = self.phase_mask_arr_files[phase]
        features = calculate_morphological_features(mask_arr_file)
        if return_as_dict:
            return features
        else:
            area = features["area"]
            perimeter = features["perimeter"]
            eccentricity = features["eccentricity"]
            axis_major_length = features["axis_major_length"]
            axis_minor_length = features["axis_minor_length"]
            return [area, perimeter, eccentricity, axis_major_length, axis_minor_length]
    
    def _get_texture_features_per_phase(self, phase, return_as_dict=True):
        if phase not in self.all_available_phases:
            return [None, None, None, None]
        image_file = self.phase_img_files[phase]
        features = calculate_texture_features(image_file)
        if return_as_dict:
            features_dict = {}
            features_dict[phase + "_contrast"] = features["contrast"]
            features_dict[phase + "_correlation"] = features["correlation"]
            features_dict[phase + "_energy"] = features["energy"]
            features_dict[phase + "_homogeneity"] = features["homogeneity"]
            return features_dict
        else:
            contrast = features["contrast"]
            correlation = features["correlation"]
            energy = features["energy"]
            homogeneity = features["homogeneity"]
            return [contrast, correlation, energy, homogeneity]

    def get_all_texture_features(self):
        texture_features = []
        for phase in self.all_available_phases:
            features = self._get_texture_features_per_phase(phase, return_as_dict=True)
            texture_features.append(features)
        return texture_features
    
    def get_mallignancy_label(self, label_csv_file):
        df = pd.read_csv(label_csv_file, sep=',')
        row = df[df["MRN"] == int(self.MRN)]
        if row.empty:
                return None
        else:
            return int(row["malignancy"].values[0])

######################################################
def get_all_MRNs_StudyDates(Instance_data_dir):
    feature_files = glob.glob(os.path.join(Instance_data_dir, "*.csv"))
    MRN_StudyDate_list = []
    for fn in feature_files:
        base_fn = os.path.basename(fn)
        p_fn = os.path.splitext(base_fn)[0]
        ele = p_fn.split('_')
        MRN = ele[0]
        StudyDate = ele[1]
        MRN_StudyDate_list.append((MRN, StudyDate))
    return MRN_StudyDate_list

def get_group_name_from_available_phases(available_phases):
    phases_for_grouping = list(set(available_phases))
    phases_for_grouping.sort(key=lambda x: PHASE_ORDER.index(x))
    from group_features import Group_Dict
    phase_grouping_str = "; ".join(phases_for_grouping)
    for group_key, group_value in Group_Dict.items():
        if group_value == phase_grouping_str:
            return group_key, group_value
    return None, None

######################################################
# Run this script to get all the features for all the cases using AdrenalInstance class
######################################################
if __name__ == "__main__":
    from group_features import *
    from config import DATA_ROOT_DIR
    processed_data_root_dir = os.path.join(DATA_ROOT_DIR, "processed_data")
    clinical_data_file = os.path.join(processed_data_root_dir, "mrn_malignancy.csv")
    HU_data_dir = os.path.join(processed_data_root_dir, "HU_array_adjusted")
    Instance_data_dir = os.path.join(processed_data_root_dir, "split_pixel_data_adjusted")
    Image_data_dir = os.path.join(processed_data_root_dir, "reconstructed_img_adjusted")
    # Output directory for saving feature CSV files
    Feature_data_dir = os.path.join(processed_data_root_dir, "grouped_instance_features")

    MRN_StudyDate_list = get_all_MRNs_StudyDates(Instance_data_dir)

    # clear existing feature csv files in the output directory
    existing_feature_files = glob.glob(os.path.join(Feature_data_dir, "Group*_features.csv"))
    for f in existing_feature_files:
        os.remove(f)

    # recored ungrouped instances
    multiple_delay_phases_instances = []
    ungrouped_instances = []

    for MRN, StudyDate in MRN_StudyDate_list:
        adrenal_instance = AdrenalInstance(MRN=MRN, StudyDate=StudyDate, HU_data_dir=HU_data_dir, Image_data_dir=Image_data_dir)
        print("Available phases:", adrenal_instance.all_available_phases)
        delay_count = adrenal_instance.all_available_phases.count("Delay")
        if delay_count > 1:
            available_phase_str = "; ".join(adrenal_instance.all_available_phases)
            case_malignancy = adrenal_instance.get_mallignancy_label(clinical_data_file)
            multiple_delay_phases_instances.append((MRN, StudyDate, delay_count, available_phase_str, case_malignancy))

        case_malignancy = adrenal_instance.get_mallignancy_label(clinical_data_file)
        print("Malignancy label:", case_malignancy)
        phase_for_morph = adrenal_instance.all_available_phases[0]
        morph_features = adrenal_instance.get_morphological_features(phase_for_morph, return_as_dict=True)
        print(f"Morphological features:", morph_features)
        avg_HU_features = adrenal_instance.calculate_Average_HU()
        print("Average HU features:", avg_HU_features)
        if adrenal_instance.is_washout_available():
            attenuation_features = adrenal_instance.get_attenuation_features()
            print("Attenuation features:", attenuation_features)
        all_texture_features = adrenal_instance.get_all_texture_features()
        print("Texture features for all available phases:", all_texture_features)
        # for phase in adrenal_instance.all_available_phases:
        #     print(f"Texture features for {phase}:", adrenal_instance._get_texture_features_per_phase(phase))

        # Determine group name
        available_phase_str = "; ".join(adrenal_instance.all_available_phases)
        # get key of Group_Dict based on the value of Group_Dict
        group_key, group_value = get_group_name_from_available_phases(adrenal_instance.all_available_phases)
        if group_key is None:   
            print(f"Warning: No group found for available phases: {available_phase_str}")
            ungrouped_instances.append((MRN, StudyDate, available_phase_str))
            continue
        print(f"Instance belongs to {group_key} with phases: {available_phase_str}")
        # Save features to CSV
        feature_output_file = os.path.join(Feature_data_dir, f"{group_key}_features.csv")
        os.makedirs(Feature_data_dir, exist_ok=True)
        feature_dict = {"MRN": MRN,
                        "StudyDate": StudyDate,
                        "group": group_key,
                        "available_phases": available_phase_str,
                        "malignancy": case_malignancy
                        }
        # Morphological features
        for feat_name, feat_value in morph_features.items():
            feature_dict[f"{feat_name}"] = feat_value
        # Average HU features
        avg_HU_features = adrenal_instance.calculate_Average_HU()
        for phase, avg_HU in avg_HU_features.items():
            feature_dict[f"avg_HU_{phase}"] = avg_HU
        # Attenuation features
        if adrenal_instance.is_washout_available():
            attenuation_feature_names = ["absolute_washout", "relative_washout", "absolute_washout_rate", "relative_washout_rate"]
            for idx, feat_name in enumerate(attenuation_feature_names):
                feature_dict[f"{feat_name}"] = attenuation_features[feat_name]
        # Texture features
        for texture_feat in all_texture_features:
            for feat_name, feat_value in texture_feat.items():
                feature_dict[f"{feat_name}"] = feat_value
        feature_df = pd.DataFrame([feature_dict])
        if os.path.exists(feature_output_file):
            feature_df.to_csv(feature_output_file, mode='a', header=False, index=False)
        else:
            feature_df.to_csv(feature_output_file, mode='w', header=True, index=False)

    print("========================================")
    print("Ungrouped instances (no matching group found):")
    for MRN, StudyDate, available_phase_str in ungrouped_instances:
        print(f"MRN: {MRN}, StudyDate: {StudyDate}, Available Phases: {available_phase_str}")
    ######################################################


    print("========================================")
    print("Instances with multiple 'Delay' phases: N=", len(multiple_delay_phases_instances))
    # save the instances with multiple delay phases together with features into a csv file
    multiple_delay_phases_df = pd.DataFrame(multiple_delay_phases_instances, columns=["MRN", "StudyDate", "Delay_count", "Available_phases", "Malignancy"])
    output_dir = os.path.join(processed_data_root_dir, "attenuation_eval")
    multiple_delay_phases_df.to_csv(os.path.join(output_dir, "instances_multiple_delay_phases.csv"), index=False)
    for MRN, StudyDate, delay_count, available_phases, malignancy in multiple_delay_phases_instances:
        print(f"MRN: {MRN}, StudyDate: {StudyDate}, Delay Count: {delay_count}, Available Phases: {available_phases}, Malignancy: {malignancy}")

