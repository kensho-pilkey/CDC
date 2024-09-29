# Carolina Data Challenge 2024 - Health Sciences Track

## Team: Analysis on Influenza, Pneumonia, and COVID Trends & Travel

Welcome to our project repository for the **Carolina Data Challenge 2024**! We are a team competing in the **Health Sciences Track**, aiming to uncover insights into influenza, pneumonia, and COVID-19 trends & travel restrictions across the United States.

## Project Overview

### **Objective**
Our primary goal is to analyze and identify patterns in **Total Pneumonia, Influenza, and COVID-19 Deaths** across different states over time. By leveraging **K-Means Time Clustering**, we aim to group states with similar mortality trends, providing valuable insights into regional health dynamics.

### **Methodology**

#### **K-Means Time Clustering**
K-Means Time Clustering is an extension of the traditional K-Means algorithm tailored for time series data. Unlike standard K-Means, which clusters based on spatial proximity, K-Means Time Clustering accounts for the temporal dimension, allowing us to group states based on the similarity of their death trends over time.

**Why K-Means Time Clustering?**
- **Suitability for Time Series Data:** Time series data involves sequences of data points indexed in time order. Traditional clustering methods may overlook the temporal dependencies and patterns inherent in such data. K-Means Time Clustering addresses this by considering the entire temporal trajectory of each state, ensuring that states with similar trends are grouped together.
- **Scalability:** K-Means is computationally efficient and scalable to large datasets, making it ideal for analyzing extensive time series data across all US states.
- **Interpretability:** The resulting clusters are straightforward to interpret, facilitating easy identification of regional patterns and trends.

#### **Evaluation Metric**
To evaluate the effectiveness of our clustering, we utilized the **Silhouette Score**, which measures how similar an object is to its own cluster compared to other clusters. A higher Silhouette Score indicates better-defined clusters with clear boundaries.

### **Key Findings**

1. **Regional Clustering Trends**
   - **Clusters Identified:**
     - **Southeast, Southwest, and Southeast Clusters:** States within these regions exhibited **similar trends** in total deaths over time, characterized by **consistent spikes and declines** corresponding to pandemic waves and seasonal respiratory illnesses.
     - **Midwest and Northeast Clusters:** States in these regions showed distinct mortality trends, differing notably from those in the Southeast and Southwest. These differences may be attributed to **varying public health responses**, **population densities**, and **socio-economic factors**.
   
   - **Geospatial Visualization:**
     - A **geospatial chart** illustrating the clustering of states is available in the [`rileykmeans.ipynb`](./rileykmeans.ipynb) notebook. This map provides a visual representation of how states are grouped based on their mortality trends.

2. **Correlation with Flight Cancellations**
   - Utilizing a **separate dataset** on flight cancellations, our analysis revealed that the **start of 2020** experienced a **significant spike in cancellations**. This spike closely **mirrored the surge in COVID-19 deaths**, highlighting the impact of the **initial outbreak** and the implementation of **strict travel restrictions** globally.
   - Additionally, this period saw **increases in pneumonia and influenza deaths**, particularly during the **winter months**, which traditionally experience peaks in respiratory illnesses. The alignment of these trends underscores the interplay between **public health measures**, **virus transmission**, and **seasonal factors**.

### **Conclusion**
Our analysis demonstrates the effectiveness of **K-Means Time Clustering** in identifying and visualizing regional patterns in health-related mortality data. By clustering states based on their temporal death trends, we uncover significant regional similarities and differences, providing a foundation for targeted public health interventions and policy-making.

The correlation between flight cancellations and COVID-19 deaths further emphasizes the interconnectedness of **public health responses** and **mobility patterns**, offering insights into how global events can influence local health outcomes.

## Repository Structure

- **`rileykmeans.ipynb`**: Contains the implementation of K-Means Time Clustering, geospatial visualizations, and detailed analysis of clustering results.
- **`kensho.ipynb`**: Explores the relationship between flight cancellations and COVID-19 deaths, highlighting significant temporal correlations.

### **Installation**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/carolina-data-challenge.git
   cd carolina-data-challenge
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Analysis**
1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open and Run Notebooks**
   - Navigate to [`rileykmeans.ipynb`](./rileykmeans.ipynb) to explore the clustering analysis and geospatial visualizations.
   - Open [`kensho.ipynb`](./flight_cancellations_analysis.ipynb) to examine the correlation between flight cancellations and COVID-19 deaths.

## Acknowledgements
We would like to thank the organizers of the **Carolina Data Challenge** for providing the platform to showcase our analytical skills and contribute to meaningful research in the health sciences domain.

## Contact

Riley Harper - riley.harper@unc.edu

Kensho Pilkey - kpilkey@unc.edu

Aidan Guenthner - aidguent@unc.edu

---
