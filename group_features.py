
PHASES = ["NC", "AR", "PV", "Delay"]
META_COLS = ["MRN", "StudyDate", "group", "available_phases", "malignancy"]

Group_Dict = {"Group1": "NC",
               "Group2": "AR",
               "Group3": "PV",
               "Group4": "Delay",
               "Group5": "NC; AR; PV; Delay",
               "Group6": "AR; PV; Delay",
               "Group7": "NC; PV; Delay",
               "Group8": "NC; AR; Delay",
               "Group9": "NC; AR; PV",
               "Group10": "PV; Delay",
               "Group11": "NC; Delay",
               "Group12": "NC; AR",
               "Group13": "AR; Delay",
               "Group14": "NC; PV",
               "Group15": "AR; PV"}


morph_features = ["area", "perimeter", "eccentricity", "axis_major_length", "axis_minor_length"]
atten_features = ["absolute_washout", "relative_washout", "absolute_washout_rate", "relative_washout_rate"]

ALL_FEATURES_FOR_DATA_FRAME =  morph_features + atten_features + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity","avg_HU_NC",
                                                      "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity","avg_HU_AR",
                                                      "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity","avg_HU_PV",
                                                      "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity","avg_HU_Delay"
                                                      ]

ALL_FEATURES_FOR_DATA_FRAME_EXCLUDE_WASHOUT = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_PV", "avg_HU_Delay"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                                      "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                                      "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                                      "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                                      ]

Group1_features = morph_features + ["avg_HU_NC"] +["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity"]
Group2_features = morph_features + ["avg_HU_AR"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity"]
Group3_features = morph_features + ["avg_HU_PV"] + ["PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"]
Group4_features = morph_features + ["avg_HU_Delay"] + ["Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"]
Group5_features = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_PV", "avg_HU_Delay"] + atten_features + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                                      "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                                      "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                                      "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                                      ]
Group6_features = morph_features + ["avg_HU_AR", "avg_HU_PV", "avg_HU_Delay"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]
Group7_features = morph_features + ["avg_HU_NC", "avg_HU_PV", "avg_HU_Delay"] + atten_features + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]
Group8_features = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_Delay"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]
Group9_features = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_PV"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"
                                    ]
Group10_features = morph_features + ["avg_HU_PV", "avg_HU_Delay"] + ["PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]


Group11_features = morph_features + ["avg_HU_NC", "avg_HU_Delay"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]

Group12_features = morph_features + ["avg_HU_NC", "avg_HU_AR"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity"
                                    ]

Group13_features = morph_features + ["avg_HU_AR", "avg_HU_Delay"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]


Group14_features = morph_features + ["avg_HU_NC", "avg_HU_PV"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"
                                    ]


Group15_features = morph_features + ["avg_HU_AR", "avg_HU_PV"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"
                                    ]


ALL_FEATURES_DISPLAY_NAMES = ["NC_area", "NC_perimeter", "NC_eccentricity", "NC_axis_major_length", "NC_axis_minor_length", "avg_HU_NC",
                "NC_absolute_washout", "NC_relative_washout", "NC_absolute_washout_rate", "NC_relative_washout_rate",
                "NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                "AR_area", "AR_perimeter", "AR_eccentricity", "AR_axis_major_length", "AR_axis_minor_length", "avg_HU_AR",
                "AR_absolute_washout", "AR_relative_washout", "AR_absolute_washout_rate", "AR_relative_washout_rate",
                "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                "PV_area", "PV_perimeter", "PV_eccentricity", "PV_axis_major_length", "PV_axis_minor_length", "avg_HU_PV",
                "PV_absolute_washout", "PV_relative_washout", "PV_absolute_washout_rate", "PV_relative_washout_rate",
                "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                "Delay_area", "Delay_perimeter", "Delay_eccentricity", "Delay_axis_major_length", "Delay_axis_minor_length", "avg_HU_Delay",    
                "Delay_absolute_washout", "Delay_relative_washout", "Delay_absolute_washout_rate", "Delay_relative_washout_rate",
                "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
               ]
