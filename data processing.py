import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# First, load your data
train_data = pd.read_csv('train.csv')


# Let's organize all our analysis functions in a class for better structure
class AlzheimerDataAnalyzer:
    def __init__(self, data):
        self.data = data

    def basic_data_exploration(self):
        """
        Performs initial data exploration and prints basic statistics
        """
        print("=== Basic Data Exploration ===")
        print("\nDataset Dimensions:", self.data.shape)
        print("\nFeature Types:\n", self.data.dtypes)
        print("\nMissing Values:\n", self.data.isnull().sum())
        print("\nNumerical Features Summary:\n", self.data.describe())

    def analyze_demographics(self):
        """
        Creates visualizations of demographic distributions across different categories
        including age, gender, education level, and diagnosis status.

        Requirements:
        - self.data should be a pandas DataFrame containing columns:
            - 'Age': numeric values
            - 'Gender': categorical
            - 'EducationLevel': categorical
            - 'Diagnosis': binary (0 or 1)
        """

        # Set the style for better visualization
        sns.set_style("whitegrid")

        # Create a figure with adequate size and spacing
        plt.figure(figsize=(15, 8))

        # First, convert the Diagnosis column to standard Python integers
        # This ensures consistent behavior across different plot types
        self.data['Diagnosis'] = self.data['Diagnosis'].astype(int)

        # Define color palette using integer keys to match the data type
        diagnosis_palette = {0: 'lightblue', 1: 'salmon'}

        # 1. Age distribution visualization
        plt.subplot(141)
        sns.histplot(
            data=self.data,
            x='Age',
            hue='Diagnosis',
            multiple="stack",
            palette=diagnosis_palette
        )
        plt.title('Age Distribution by Diagnosis', pad=15)

        # 2. Gender distribution visualization
        plt.subplot(142)
        sns.countplot(
            data=self.data,
            x='Gender',
            hue='Diagnosis',
            palette=diagnosis_palette
        )
        plt.title('Gender Distribution by Diagnosis', pad=15)

        # 3. Education level distribution visualization
        plt.subplot(143)
        sns.countplot(
            data=self.data,
            x='EducationLevel',
            hue='Diagnosis',
            palette=diagnosis_palette
        )
        plt.xticks(rotation=45, ha='right')
        plt.title('Education Level by Diagnosis', pad=15)

        # 4. Overall diagnosis distribution
        plt.subplot(144)
        sns.countplot(
            data=self.data,
            x='Diagnosis',
            hue='Diagnosis',
            palette=diagnosis_palette,
            legend=False
        )
        plt.title('Diagnosis Distribution', pad=15)
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig("demographics_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()
    def analyze_clinical_measurements(self):
        """
        Analyzes distribution of clinical measurements
        """
        print("\n=== Clinical Measurements Analysis ===")
        clinical_vars = ['BMI', 'SystolicBP', 'DiastolicBP',
                         'CholesterolTotal', 'CholesterolLDL',
                         'CholesterolHDL', 'CholesterolTriglycerides']

        plt.figure(figsize=(15, 10))
        for i, var in enumerate(clinical_vars, 1):
            plt.subplot(2, 4, i)
            sns.boxplot(data=self.data, x='Diagnosis', y=var)
            plt.xticks(rotation=45)
            plt.title(f'{var} by Diagnosis')
        plt.tight_layout()
        plt.savefig("clinical_measurements.png", dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_cognitive_functional(self):
        """
        Analyzes cognitive and functional assessments
        """
        print("\n=== Cognitive and Functional Analysis ===")
        cognitive_vars = ['MMSE', 'FunctionalAssessment',
                          'MemoryComplaints', 'BehavioralProblems', 'ADL']

        plt.figure(figsize=(15, 5))
        for i, var in enumerate(cognitive_vars, 1):
            plt.subplot(1, 5, i)
            sns.violinplot(data=self.data, x='Diagnosis', y=var)
            plt.xticks(rotation=45)
            plt.title(f'{var} Distribution')
        plt.tight_layout()
        plt.savefig("cognitive_functional.png", dpi=300, bbox_inches="tight")
        plt.show()

    def correlation_analysis(self):
        """
        Performs correlation analysis on numerical features
        """
        print("\n=== Correlation Analysis ===")
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        correlations = self.data[numerical_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig("correlation_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()


# Now, let's use these functions:
def main():
    # Load the data
    print("Loading data...")
    train_data = pd.read_csv('train.csv')

    # Create analyzer object
    analyzer = AlzheimerDataAnalyzer(train_data)

    # Run all analyses
    analyzer.basic_data_exploration()
    analyzer.analyze_demographics()
    analyzer.analyze_clinical_measurements()
    analyzer.analyze_cognitive_functional()
    analyzer.correlation_analysis()


# Run the analysis
if __name__ == "__main__":
    main()