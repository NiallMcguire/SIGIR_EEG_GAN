import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
import math


class DataSetup:

    def __init__(self):
        pass

    def load_features(self, file_path):
        FlightConditionNames = [r"\Baseline_Subject_Dictionary.json",
                                r"\SnL_Subject_Dictionary.json",
                                r"\TwoG_Subject_Dictionary.json"]

        FLightConditionData = []
        # Opening JSON file
        for FlightConditionName in FlightConditionNames:
            with open(file_path + FlightConditionName) as json_file:
                data = json.load(json_file)
                FLightConditionData.append(data)

        return FLightConditionData


    def create_rolling_window(self, data, window_size=4):
        """
        This function creates a rolling window of a given size from a data frame.
        It assigns the majority label to the window.

        Creator : Niall

        Args:
            data: Dataframe to be windowed
            window_size: The size of the window to be created

        Returns:
            windows: A list of dataframes containing the windows
            labels: A list of labels corresponding to the windows

        """

        windows = []
        labels = []
        data = data.to_numpy()
        for i in range(len(data) - window_size + 1):
            window_data = data[i:i + window_size]
            label = window_data[:, -1][0]  # Assign majority label
            labels.append(label)
            window_data = window_data[:, :-1]
            windows.append(window_data)

        return windows, labels

    def AlignFlightTrials(self, Phase, FlightConditionDict):
        """
        This function takes in a dictionary of a given flight condition and orders each of the trials in a sequential order

        Creator : Niall

        Args:
            FlightConditionDict: A dictionary containing the flight condition data

        Returns:
            CombinationDict: A dictionary containing the flight condition data with the trials ordered sequentially for each participant

        """

        CombinationDict = {}
        Participants = self.participant_list

        for participant in Participants:
            CombinedSequenceList = []
            CombinedLabelList = []

            TrialOne = FlightConditionDict[participant]["T1"]
            TrialTwo = FlightConditionDict[participant]["T2"]
            TrialThree = FlightConditionDict[participant]["T3"]

            TrialOne = pd.DataFrame.from_dict(TrialOne)
            TrialTwo = pd.DataFrame.from_dict(TrialTwo)
            TrialThree = pd.DataFrame.from_dict(TrialThree)

            if Phase == 1:
                TrialOne["Label"] = 0
                TrialTwo["Label"] = 1
                TrialThree["Label"] = 2
            elif Phase == 2:
                TrialOne["Label"] = 0
                TrialTwo["Label"] = 0
                TrialThree["Label"] = 0

            TrialOne = TrialOne.drop(columns=["TimeStamp"])
            TrialTwo = TrialTwo.drop(columns=["TimeStamp"])
            TrialThree = TrialThree.drop(columns=["TimeStamp"])

            TrialOneWindow, TrialOneLabel = self.create_rolling_window(TrialOne)
            TrialTwoWindow, TrialTwoLabel = self.create_rolling_window(TrialTwo)
            TrialThreeWindow, TrialThreeLabel = self.create_rolling_window(TrialThree)

            for i in range(0, len(TrialOneWindow)):
                CombinedSequenceList.append(TrialOneWindow[i])
                CombinedSequenceList.append(TrialTwoWindow[i])
                CombinedSequenceList.append(TrialThreeWindow[i])

                CombinedLabelList.append(TrialOneLabel[i])
                CombinedLabelList.append(TrialTwoLabel[i])
                CombinedLabelList.append(TrialThreeLabel[i])

            CombinationDict[participant] = {"Windows": CombinedSequenceList, "Label": CombinedLabelList}

        return CombinationDict

    def CombineFlightConditions(self, Phase, Baseline, Snl, TwoG):
        """
        This function takes in the order flight conditions and orders them sequentially for each participant

        Creator : Niall

        Args:
            Baseline: A dictionary containing the baseline flight condition data
            Snl: A dictionary containing the SnL flight condition data

        Returns:
            CombinationDict: A dictionary containing the combined flight condition data

        """

        CombinationDict = {}
        Participants = self.participant_list
        for participant in Participants:
            CombinedFlightWindows = []
            CombinedFlightLabels = []

            Start = 0
            End = 3

            if Phase == 1:
                BaselineWindow = Baseline[participant]['Windows']
                BaselineLabel = Baseline[participant]['Label']

                SnlWindow = Snl[participant]['Windows']
                SnlLabel = Snl[participant]['Label']

                TwoGWindow = TwoG[participant]['Windows']
                TwoGLabel = TwoG[participant]['Label']

                while End <= len(BaselineWindow):
                    #Appending Window Segments
                    BaselineWindowSegment = BaselineWindow[Start:End]
                    for x in BaselineWindowSegment:
                        CombinedFlightWindows.append(x)
                    SnlWindowSegment = SnlWindow[Start:End]
                    for x in SnlWindowSegment:
                        CombinedFlightWindows.append(x)
                    TwoGWindowSegment = TwoGWindow[Start:End]
                    for x in TwoGWindowSegment:
                        CombinedFlightWindows.append(x)

                    #Appending Label Segments
                    BaselineLabelSegment = BaselineLabel[Start:End]
                    for x in BaselineLabelSegment:
                        CombinedFlightLabels.append(x)
                    SnlLabelSegment = SnlLabel[Start:End]
                    for x in SnlLabelSegment:
                        CombinedFlightLabels.append(x)
                    TwoGLabelSegment = TwoGLabel[Start:End]
                    for x in TwoGLabelSegment:
                        CombinedFlightLabels.append(x)
                    Start += 3
                    End += 3

            elif Phase == 2:
                '''
                HWWindow = HW[participant]['Windows']
                HWLabel = HW[participant]['Label']

                LWWindow = LW[participant]['Windows']
                LWLabel = LW[participant]['Label']

                while End <= len(HWWindow):
                    #Appending Window Segments
                    HWWindowSegment = HWWindow[Start:End]
                    for x in HWWindowSegment:
                        CombinedFlightWindows.append(x)
                    LWWindowSegment = LWWindow[Start:End]
                    for x in LWWindowSegment:
                        CombinedFlightWindows.append(x)

                    #Appending Label Segments
                    HWLabelSegment = HWLabel[Start:End]
                    for x in HWLabelSegment:
                        CombinedFlightLabels.append(x)
                    LWLabelSegment = LWLabel[Start:End]
                    for x in LWLabelSegment:
                        CombinedFlightLabels.append(x)
                    Start += 3
                    End += 3
                '''
            CombinationDict[participant] = {"Windows": CombinedFlightWindows, "Label": CombinedFlightLabels}
        return CombinationDict

    def LoadDataSet(self, Phase, file_path):
        """
        Args:
            Phase: Phase dictates if the data has 3 classes or 2 classes respective to the flight trial conditions

        Returns: A dictionary with participants as the key, and the windowed data and labels as the values

        """

        FlightConditionData = self.load_features(file_path)
        Baseline = self.AlignFlightTrials(Phase,FlightConditionData[0])
        SnL = self.AlignFlightTrials(Phase,FlightConditionData[1])
        TwoG = self.AlignFlightTrials(Phase,FlightConditionData[2])

        #LW = self.AlignFlightTrials(Phase,FlightConditionData[3])
        #HW = self.AlignFlightTrials(Phase,FlightConditionData[4])

        CombinedFlight = self.CombineFlightConditions(Phase, Baseline, SnL, TwoG)
        return CombinedFlight


    def TSCV_validation(self, ParticipantDictionary):
        Windows = ParticipantDictionary["Windows"]
        Labels = ParticipantDictionary["Label"]

        Windows = np.array(Windows)
        Labels = np.array(Labels)

        TSCV_Folds = {}
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
        for i, (train_index, test_index) in enumerate(tscv.split(Windows)):
            print(f"Fold {i}:")

            validation_length = math.floor(len(train_index) / 100 * 20)
            validation_index = train_index[-validation_length:]
            train_index = train_index[:-validation_length]
            # print(f"  Train: index={train_index}")
            # print(f"  Val:  index={validation_index}")
            # print(f"  Test:  index={test_index}")

            X_Train = Windows[train_index]
            X_Val = Windows[validation_index]
            X_Test = Windows[test_index]

            Y_Train = Labels[train_index]
            Y_Val = Labels[validation_index]
            Y_Test = Labels[test_index]

            TSCV_Folds[i] = [X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test]
        return TSCV_Folds

    def Kfold_validation(self, ParticipantDictionary):
        Windows = ParticipantDictionary["Windows"]
        Labels = ParticipantDictionary["Label"]

        Windows = np.array(Windows)
        Labels = np.array(Labels)

        Kfold_Folds = {}
        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(Windows)):
            print(f"Fold {i}:")
            validation_length = math.floor(len(train_index) / 100 * 20)
            validation_index = train_index[-validation_length:]
            train_index = train_index[:-validation_length]
            # print(f"  Train: index={train_index}")
            # print(f"  Val:  index={validation_index}")
            # print(f"  Test:  index={test_index}")

            X_Train = Windows[train_index]
            X_Val = Windows[validation_index]
            X_Test = Windows[test_index]

            Y_Train = Labels[train_index]
            Y_Val = Labels[validation_index]
            Y_Test = Labels[test_index]

            Kfold_Folds[i] = [X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test]
        return Kfold_Folds








