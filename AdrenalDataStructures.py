
import numpy as np
import pandas as pd
import os
from PIL import Image
from datetime import datetime

class AdrenalPhaseDataAtDelataT:
    def __init__(self, phase_delta_t_df): 
        self.MRN = str(list(phase_delta_t_df["MRN"])[0])
        self.StudyDate = str(list(phase_delta_t_df["DATE"])[0])
        self.pixel_data = phase_delta_t_df["pixel_value"]
        self.L_R = phase_delta_t_df["LEFT_or_RIGHT"]
        self.phase = (phase_delta_t_df["Phase"]).unique()[0]
        self.delta_t = phase_delta_t_df["delta_time(s)"].unique()[0] 
        self.pixel_coord_x = phase_delta_t_df["x"]
        self.pixel_coord_y = phase_delta_t_df["y"]
        self.slice_thickness = phase_delta_t_df["SLICE_THICKNESS"].unique()[0] # can be multiple values
        self.Lesion_Number = phase_delta_t_df["Lesion_No"].unique()[0]

    def is_both_sides_lesions(self):
        set_L_R = set(self.L_R)
        if len(set_L_R) == 0:
            raise ValueError("No side found in the dataframe.")
        elif len(set_L_R) == 1:
            return False
        else:
            return True
    
    def HU_window(self, pixel_list, center=40, width=400):
        """Apply HU windowing to CT image. Use Soft tissue window""" 
        lower = center - width // 2
        upper = center + width // 2
        # clipped_pixel_arr = np.clip(image, lower, upper)
        # return clipped_pixel_arr
        clipped_pixel = [min(max(x, lower), upper) for x in pixel_list]
        return clipped_pixel

    # TODO: better way to create jpeg image
    def create_PIL_img(self, img_size=512, side="B", center=True):
        if len(self.pixel_data) == 0:
            raise ValueError("No pixel data available for the specified side.")
        else:
            if side == "LT":
                pixel_coord_x = self.pixel_coord_x[self.L_R == "LT"]
                pixel_coord_y = self.pixel_coord_y[self.L_R == "LT"]
                pixel_data = self.pixel_data[self.L_R == "LT"]
            elif side == "RT":
                pixel_coord_x = self.pixel_coord_x[self.L_R == "RT"]
                pixel_coord_y = self.pixel_coord_y[self.L_R == "RT"]
                pixel_data = self.pixel_data[self.L_R == "RT"]
            elif side == "B":
                pixel_coord_x = self.pixel_coord_x
                pixel_coord_y = self.pixel_coord_y
                pixel_data = self.pixel_data
            else:
                raise ValueError("Invalid side specified. Use 'LT', 'RT', or 'B'.")
            # put the pixel data to the center of the image
            if center:
                # Calculate the bounding box of the coordinates
                min_x, max_x = pixel_coord_x.min(), pixel_coord_x.max()
                min_y, max_y = pixel_coord_y.min(), pixel_coord_y.max()
                width, height = max_x - min_x + 1, max_y - min_y + 1
                
                # Calculate the offsets to center the image
                offset_x = (img_size - width) // 2 - min_x
                offset_y = (img_size - height) // 2 - min_y
                
                pixel_coord_x += offset_x
                pixel_coord_y += offset_y

            windowed_pixel_list = self.HU_window(pixel_data, center=40, width=400) # use soft tissue window
            windowed_pixel_array = np.array(windowed_pixel_list)
            normalized = (windowed_pixel_array - windowed_pixel_array.min()) / (windowed_pixel_array.max() - windowed_pixel_array.min())
            normalized = (normalized * 255)
            img_arr = np.zeros((img_size, img_size), dtype=np.uint8)
            for x, y, value in zip(pixel_coord_x, pixel_coord_y, normalized):
                if x == 512 or y == 512:
                    print(f"Pixel coordinates out of bounds: x={x}, y={y}")
                img_arr[y, x] = value
            # normalized = (img_arr - min(windowed_pixel_list)) / (max(windowed_pixel_list) - min(windowed_pixel_list))
            # normalized = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
            # normalized = (normalized * 255).astype(np.uint8)  # scale to [0,255] for PIL compatibility
            img = Image.fromarray(img_arr)
            img = img.convert("RGB")  # replicates to 3 channels
            # img = img.convert("L")  # convert to grayscale
            return img

    def save_PIL_img(self, img_size=512, save_dir=None, img_name=None):
        if save_dir is None:
            raise ValueError("Save directory is not specified.")
        if img_name is None:
            raise ValueError("Image name is not specified.")
        PIL_img = self.create_PIL_img(img_size)
        img_path = os.path.join(save_dir, img_name)
        PIL_img.save(img_path)
        # Image.fromarray(PIL_img).convert("L").save(img_path)
        print(f"Image saved at {img_path}")

    def create_HU_array(self, arr_size=512, side="B", center=True):
        if len(self.pixel_data) == 0:
            raise ValueError("No pixel data available for the specified side.")
        else:
            if side == "LT":
                pixel_coord_x = self.pixel_coord_x[self.L_R == "LT"]
                pixel_coord_y = self.pixel_coord_y[self.L_R == "LT"]
                pixel_data = self.pixel_data[self.L_R == "LT"]
            elif side == "RT":
                pixel_coord_x = self.pixel_coord_x[self.L_R == "RT"]
                pixel_coord_y = self.pixel_coord_y[self.L_R == "RT"]
                pixel_data = self.pixel_data[self.L_R == "RT"]
            elif side == "B":
                pixel_coord_x = self.pixel_coord_x
                pixel_coord_y = self.pixel_coord_y
                pixel_data = self.pixel_data
            else:
                raise ValueError("Invalid side specified. Use 'LT', 'RT', or 'B'.")
            # put the pixel data to the center of the image
            if center:
                # Calculate the bounding box of the coordinates
                min_x, max_x = pixel_coord_x.min(), pixel_coord_x.max()
                min_y, max_y = pixel_coord_y.min(), pixel_coord_y.max()
                width, height = max_x - min_x + 1, max_y - min_y + 1
                
                # Calculate the offsets to center the image
                offset_x = (arr_size - width) // 2 - min_x
                offset_y = (arr_size - height) // 2 - min_y
                
                pixel_coord_x += offset_x
                pixel_coord_y += offset_y

            # HU_img_arr = np.zeros((arr_size, arr_size), dtype=np.uint8)  # wrong datatype !!!
            HU_img_arr = np.zeros((arr_size, arr_size), dtype=np.int16)
            mask_arr = np.zeros((arr_size, arr_size), dtype=bool)
            for x, y, value in zip(pixel_coord_x, pixel_coord_y, pixel_data):
                if x == 512 or y == 512:
                    print(f"Pixel coordinates out of bounds: x={x}, y={y}")
                HU_img_arr[y, x] = value
                mask_arr[y, x] = True
            return HU_img_arr, mask_arr

    def save_img_by_lesion_side(self, img_size=512, save_dir=None, img_name=None, split_LR=True):
        if save_dir is None:
            raise ValueError("Save directory is not specified.")
        if img_name is None:
            raise ValueError("Image name is not specified.")
        is_both_sides_lesion = self.is_both_sides_lesions()
        
        if is_both_sides_lesion == True and split_LR==False:
            img_B = self.create_PIL_img(img_size, side="B")
            img_B_path = os.path.join(save_dir, img_name.replace(".jpg", "_B.jpg"))
            img_B.save(img_B_path)
            print(f"Both side image saved at {img_B_path}")
        else:
            lesion_side = self.L_R.unique()[0]
            img_path = os.path.join(save_dir, img_name.replace(".jpg", f"_{lesion_side}.jpg"))
            img = self.create_PIL_img(img_size, side=lesion_side)
            img.save(img_path)
            print(f"{lesion_side} image saved at {img_path}")

    def save_HU_arry_by_lesion_side(self, arr_size=512, save_dir=None, fn_name=None, split_LR=True):
        if save_dir is None:
            raise ValueError("Save directory is not specified.")
        if fn_name is None:
            raise ValueError("HU arry file name is not specified.")
        is_both_sides_lesion = self.is_both_sides_lesions()
        
        if is_both_sides_lesion == True and split_LR==False:
            arr_B, mask_arr = self.create_HU_array(arr_size, side="B")
            arr_B_path = os.path.join(save_dir, fn_name.replace(".jpg", "_B.npy"))
            mask_arr_path = os.path.join(save_dir, fn_name.replace(".jpg", "_B_mask.npy"))
            np.save(arr_B_path, arr_B)
            np.save(mask_arr_path, mask_arr)
            print(f"Both side HU arry saved at {arr_B_path}")
        else:
            lesion_side = self.L_R.unique()[0]
            arr_path = os.path.join(save_dir, fn_name.replace(".jpg", f"_{lesion_side}.npy"))
            mask_arr_path = os.path.join(save_dir, fn_name.replace(".jpg", f"_{lesion_side}_mask.npy"))
            HU_arr, mask_arr = self.create_HU_array(arr_size, side=lesion_side)
            np.save(arr_path, HU_arr)
            np.save(mask_arr_path, mask_arr)
            print(f"{lesion_side} HU array saved at {arr_path}")


class AdrenalPhaseData:
    def __init__(self, phase_df):
        self.phase_df = phase_df 
        valid, message = self.validate_phase()
        if valid:
            self.phase_delta_t_data = self.split_phase_df_to_time_points_by_delta_t(phase_df)
        else:
            raise Exception(f"Error in the phase: {message}")
        
    def split_phase_df_to_time_points_by_delta_t(self, phase_df):
        phase_delta_t_data_list = [group for _, group in phase_df.groupby("delta_time(s)")]
        return phase_delta_t_data_list

    def validate_phase(self):
        valid = True
        message = "None"
        phase_name = self.phase_df["Phase"].unique()[0]
        if len(self.phase_df["delta_time(s)"].unique()) > 1 and phase_name in ["NC", "AR", "PV"]:
            valid = False
            message = f"More than one delta_t found in {phase_name}."
        return valid, message

class AdrenalStudy:
    def __init__(self, study_df):
        self.study_df = study_df

        if not self.validate_study()[0]:
            raise Exception(f"Error in the study: {self.validate_study()[1]}")
        else:
            self.phase_df_list = self.split_study_to_phases_by_phase_name()
    
    def get_MRN(self):
        MRN =self.study_df['MRN'].unique()
        assert len(MRN) == 1, "More than one MRN found in the dataframe."
        return str(MRN[0])
    
    def get_AllPhasesNames(self, phase_column_key="Phase"):
        All_phases_names = self.study_df[phase_column_key].unique()
        All_phases_names = [phase.strip() for phase in All_phases_names]
        return All_phases_names
    
    def get_all_sorted_phases_names(self):
        All_phases_names = self.get_AllPhasesNames()
        sorted_phase_names = sorted(All_phases_names, key=lambda x: ['NC', 'AR', 'PV', 'Delay'].index(x) if x in ['NC', 'AR', 'PV', 'Delay'] else len(All_phases_names))
        return sorted_phase_names

    def get_all_delta_t_in_phases(self, sorted_phase_names):
        phase_delta_t_dict = {}
        for phase in sorted_phase_names:
            delta_t = self.study_df[self.study_df["Phase"] == phase]["delta_time(s)"].unique()
            phase_delta_t_dict[phase] = delta_t
        return phase_delta_t_dict

    def get_StudyDate(self):
        StudyDate = self.study_df['DATE'].unique()
        assert len(StudyDate) == 1, "More than one StudyDate found in the dataframe."
        return str(StudyDate[0])

    def split_study_to_phases_by_phase_name(self):
        All_phases_names = self.get_AllPhasesNames()
        phase_df_list = []
        for phase in All_phases_names:
            phase_df = self.study_df[self.study_df["Phase"] == phase]
            phase_df_list.append(phase_df)
        return phase_df_list
    
    def validate_study(self):
        valid = True
        message = "None"
        # Check if MRN is consistent
        MRN =self.study_df['MRN'].unique()
        if not len(MRN) == 1:
            valid = False
            message = "More than one MRN found in the dataframe."
        # Check if StudyDate is consistent
        StudyDate = self.study_df['DATE'].unique()
        if not len(StudyDate) == 1:
            valid = False
            message = "More than one StudyDate found in the dataframe."
        return valid, message

    def get_lesion_info(self):
        lesion_info = self.study_df["LEFT_or_RIGHT"].unique()
        return lesion_info 

    def get_study_info(self):
        MRN = self.get_MRN()
        StudyDate = self.get_StudyDate()
        AllSortedPhaseNames = self.get_all_sorted_phases_names()
        AllPhaseDeltaT = self.get_all_delta_t_in_phases(AllSortedPhaseNames)
        LesionInfo = self.get_lesion_info()
        study_info = {
            "MRN": MRN,
            "StudyDate": StudyDate,
            "LesionInfo": LesionInfo,
            "Phases": AllSortedPhaseNames,
            "DeltaT": AllPhaseDeltaT
        }
        return study_info
    
    def save_study_phase_group_info(self, save_dir, study_info):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        phase_group = "_".join(study_info["Phases"])
        study_info_path = os.path.join(save_dir, f"{phase_group}_info.csv")
        if os.path.exists(study_info_path):    
            print("File already exists. Appending data.")
            df = pd.read_csv(study_info_path)
            new_row = pd.DataFrame({
                "MRN": [study_info["MRN"]],
                "StudyDate": [study_info["StudyDate"]],
                "LesionInfo": [study_info["LesionInfo"]],
                **{phase: [study_info["DeltaT"][phase]] for phase in study_info["Phases"]}
            })
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(study_info_path, index=False)
        else:
            print("File does not exist. Creating new file.")
            headers_str = "MRN,StudyDate,LesionInfo," + ",".join(study_info["Phases"])
            df = pd.DataFrame(columns=headers_str.split(","))
            df["MRN"] = [study_info["MRN"]]
            df["StudyDate"] = [study_info["StudyDate"]]
            df["LesionInfo"] = [study_info["LesionInfo"]]
            for phase in study_info["Phases"]:
                df[phase] = [study_info["DeltaT"][phase]]
            df.to_csv(study_info_path, index=False)
        print(f"Study info saved at {study_info_path}")
    

class AdrenalRawData:
    def __init__(self, text_file, sep=","):
        self.original_df = pd.read_csv(text_file, sep=sep)
        self.MRN = self.get_MRN()
        if self.more_than_one_study_date():
            print("More than one study/date found.")
            self.study_df_list = self.split_to_studies_by_date()
        else:
            self.study_df_list = [self.original_df]

    def more_than_one_study_date(self, date_key="DATE"):
        unique_dates = self.original_df[date_key].unique()
        if len(unique_dates) > 1:
            return True
        else:
            return False
    
    def get_MRN(self, data_key="MRN"):
        MRN =self.original_df[data_key].unique()
        assert len(MRN) == 1, "More than one MRN found in the dataframe."
        return str(MRN[0])

    def split_to_studies_by_date(self, date_key="DATE"):
        split_study_dfs = []
        unique_dates = self.original_df[date_key].unique()
        for date in unique_dates:
            split_study_dfs.append(self.original_df[self.original_df[date_key] == date])
        return split_study_dfs

    def save_study_date_info(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for study_df in self.study_df_list:
            study_date = study_df["DATE"].unique()[0]
            study_df.to_csv(os.path.join(save_dir, self.MRN + f"_{study_date}.csv"), index=False)



######################################################
def save_log_file(log_file, MRN, study_date, log_message):
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Log file created on {current_time}.\n")
    with open(log_file, 'a') as f:
        f.write(",".join([MRN, study_date, log_message]) + "\n")
    print(f"Log message saved at {log_file}")

######################################################
# Run this script to process and save adrenal data
######################################################
if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    # ############### Data directories before NC AR adjustment ###############
    # original_text_data_dir = os.path.join(DATA_ROOT_DIR, "original_data/data_pixels")
    # processed_data_root_dir = os.path.join(DATA_ROOT_DIR, "processed_data")
    # updated_adrenal_raw_save_dir = os.path.join(processed_data_root_dir, "pixel_data")
    # updated_adrenal_img_save_dir = os.path.join(processed_data_root_dir, "reconstructed_img")
    # grouped_study_save_dir = os.path.join(processed_data_root_dir, "grouped_study_by_phase")
    # split_updated_adrenal_raw_save_dir = os.path.join(processed_data_root_dir, "split_pixel_data")

    # ############### Data directories after NC AR adjustment ###############
    processed_data_root_dir = os.path.join(DATA_ROOT_DIR, "processed_data")
    updated_adrenal_raw_save_dir = os.path.join(processed_data_root_dir, "pixel_data_adjusted")
    updated_adrenal_img_save_dir = os.path.join(processed_data_root_dir, "reconstructed_img_adjusted")
    grouped_study_save_dir = os.path.join(processed_data_root_dir, "grouped_study_by_phase_adjusted")
    split_updated_adrenal_raw_save_dir = os.path.join(processed_data_root_dir, "split_pixel_data_adjusted")
    updated_adrenal_HU_save_dir = os.path.join(processed_data_root_dir, "HU_array_adjusted")

    if not os.path.exists(split_updated_adrenal_raw_save_dir):
        os.makedirs(split_updated_adrenal_raw_save_dir)
    if not os.path.exists(updated_adrenal_img_save_dir):
        os.makedirs(updated_adrenal_img_save_dir)
    if not os.path.exists(updated_adrenal_raw_save_dir):
        os.makedirs(updated_adrenal_raw_save_dir)
    if not os.path.exists(processed_data_root_dir):
        os.makedirs(processed_data_root_dir)
    if not os.path.exists(grouped_study_save_dir):
        os.makedirs(grouped_study_save_dir)
    if not os.path.exists(updated_adrenal_HU_save_dir):
        os.makedirs(updated_adrenal_HU_save_dir)
        
    log_file_path = os.path.join(processed_data_root_dir, "log_file.txt")

    for file in os.listdir(updated_adrenal_raw_save_dir):
        if file.endswith(".csv"):
            text_file = os.path.join(updated_adrenal_raw_save_dir, file)
            print(f"Processing {text_file}...")
            
            # 1. Raw data level
            adrenal_raw = AdrenalRawData(text_file)
            adrenal_study_df_list = adrenal_raw.study_df_list
            adrenal_raw.save_study_date_info(split_updated_adrenal_raw_save_dir)

            if len(adrenal_study_df_list) > 1:
                print(f"More than one study found in {text_file}.")
                MRN = adrenal_raw.get_MRN()
                study_date = adrenal_raw.original_df["DATE"].unique()
                save_log_file(log_file_path, MRN, str(study_date), "More than one study found.")
                # print(f"{adrenal_raw.get_MRN()} More than one study found in {text_file}.")
                
            # 2. Study by date level
            for i, study_df in enumerate(adrenal_study_df_list):
                adrenal_study = AdrenalStudy(study_df)
                phase_df_list = adrenal_study.phase_df_list
                
                study_info = adrenal_study.get_study_info()
                adrenal_study.save_study_phase_group_info(grouped_study_save_dir, study_info)
                
                # 3. Phase by phase_name level
                for j, phase_df in enumerate(phase_df_list):
                    adrenal_phase = AdrenalPhaseData(phase_df)
                    phase_delta_t_data_list = adrenal_phase.phase_delta_t_data
                    # 4. Phase delta_t level
                    for k, phase_delta_t_data in enumerate(phase_delta_t_data_list):
                        adrenal_phase_delta_t_data = AdrenalPhaseDataAtDelataT(phase_delta_t_data)
                        MRN = adrenal_phase_delta_t_data.MRN
                        study_date = adrenal_phase_delta_t_data.StudyDate
                        phase = adrenal_phase_delta_t_data.phase
                        delta_t = adrenal_phase_delta_t_data.delta_t
                        img_name=f"{MRN}_{study_date}_{phase}_{delta_t}.jpg"
                        adrenal_phase_delta_t_data.save_img_by_lesion_side(img_size=512, save_dir=updated_adrenal_img_save_dir, img_name=img_name, split_LR=False)
                        adrenal_phase_delta_t_data.save_HU_arry_by_lesion_side(arr_size=512, save_dir=updated_adrenal_HU_save_dir, fn_name=img_name, split_LR=False)
                        print(img_name)


                    # Save the study_df to a new CSV file
                    # study_df.to_csv(os.path.join(split_updated_adrenal_raw_save_dir, f"{adrenal_raw.get_MRN()}_study_{i}.csv"), index=False)

