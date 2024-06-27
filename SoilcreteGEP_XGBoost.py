import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Graphical User Interface (GUI) for predicting mechanical properties of metakolin-based soilcrete materials")
        master.configure(background="#FFDAB9")
        main_heading = tk.Label(master, text="Graphical User Interface (GUI) for predicting mechanical properties of metakolin-based soilcrete materials",
                                bg="#800000", fg="#FFDAB9", font=("Helvetica", 16, "bold"), pady=10)  # Orange background, Blue text
        main_heading.pack(side=tk.TOP, fill=tk.X)
        self.content_frame = tk.Frame(master, bg="#FFF0F5")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=25, pady=25)
        self.canvas = tk.Canvas(self.content_frame, bg="#FFF0F5")
        self.scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#FFF0F5")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.input_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.LEFT, fill="both", padx=20, pady=20, expand=True)

        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#2F4F4F",
                           fg="white", font=("Helvetica", 16, "bold"), pady=10)  # Green heading
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.G22C7 = -9.08269891183355
        self.G44C4 = 2.24629153938867
        self.G1C2 = 5.90012981395672
        self.G1C3 = -7.73787068967963
        self.G1C4 = 5.48097170934172
        self.G1C0 = -4.23557721213416
        self.G2C3 = -9.86144596697897
        self.G1C65 = 4.85091708120975
        self.G1C75 = -8.21951845885401
        self.G3C1 = 5.74607361543527
        self.G4C1 = -2.89442695010837
        self.G4C4 = 8.70615253151036
        self.G2C75 = 0.999033781994913
        self.G2C45 = 2.46667124393445
        self.G3C85 = -5.72496719260231
        self.G4C05 = 2.4462295441852
        self.G4C95 = 9.37181000149541
        self.G4C65 = 8.23416576037171
        self.G3C4 = 7.92900174962615
        self.G2C4 = -9.3387912981203
        self.G3C2 = -6.99976528519547
        self.G11C0 = 12.5815294308564
        self.create_slider("Water-to-binder ratio:", 0.40, 1.2, 0.48, 1)
        self.create_slider("Metakolin (%w/w in dry mix):", 0, 10, 0, 3)
        self.create_slider("Binder content (%w/w in dry mix):", 30, 50, 50, 5)
        self.create_slider("Superplasticizer (%w/w of binder):", 0, 2, 2, 7)
        self.create_slider("28-day Compressive Strength (MPa):", 12.21, 76.9, 39.31, 9)
        self.create_slider("Modulus of elasticity (GPa):", 1.974, 29.625, 5.979, 11)

        self.output_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both", padx=20, pady=20)
        heading = tk.Label(self.output_frame, text="Output", bg="#2F4F4F",
                   fg="white", font=("Helvetica", 16, "bold"), pady=10)  
        heading.grid(row=0, column=0, columnspan=4, pady=10)
        gep_heading = tk.Label(self.output_frame, text="Gene Expression Programming", bg="white",
                       fg="red", font=("Helvetica", 14, "bold"), pady=10)
        gep_heading.grid(row=1, column=0, columnspan=1, pady=10)
        xgboost_heading = tk.Label(self.output_frame, text="XGBoost", bg="white",
                           fg="red", font=("Helvetica", 14, "bold"), pady=10)
        xgboost_heading.grid(row=1, column=2, columnspan=1, pady=10)
        self.calculate_button_a = tk.Button(self.output_frame, text="Density", command=self.calculate_y_a,
                                          bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_a.grid(row=2, column=0, pady=10, padx=10)

        self.gep_output_text_a = tk.Text(self.output_frame, height=2, width=18)
        self.gep_output_text_a.grid(row=2, column=1, padx=10, pady=10)
        self.calculate_button_b = tk.Button(self.output_frame, text="Shrinkage", command=self.calculate_y_b,
                                                    bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_b.grid(row=3, column=0, pady=10, padx=10)
        self.gep_output_text_b = tk.Text(self.output_frame, height=2, width=18)
        self.gep_output_text_b.grid(row=3, column=1, padx=10, pady=10)
        self.calculate_button_c = tk.Button(self.output_frame, text="Strain", command=self.calculate_y_c,
                                                bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_c.grid(row=4, column=0, pady=10, padx=10)
        self.gep_output_text_c = tk.Text(self.output_frame, height=2, width=18)
        self.gep_output_text_c.grid(row=4, column=1, padx=10, pady=10)
        self.xgboost_button_aa = tk.Button(self.output_frame, text="Density", command=self.calculate_xgboost_aa,
                                                bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.xgboost_button_aa.grid(row=2, column=2, pady=10, padx=10)

        self.xgboost_output_text_aa = tk.Text(self.output_frame, height=2, width=18)
        self.xgboost_output_text_aa.grid(row=2, column=3, padx=10, pady=10)
        self.xgboost_button_bb = tk.Button(self.output_frame, text="Shrinkage", command=self.calculate_xgboost_bb,
                                                bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.xgboost_button_bb.grid(row=3, column=2, pady=10, padx=10)
        self.xgboost_output_text_bb = tk.Text(self.output_frame, height=2, width=18)
        self.xgboost_output_text_bb.grid(row=3, column=3, padx=10, pady=10)
        self.xgboost_button_cc = tk.Button(self.output_frame, text="Strain", command=self.calculate_xgboost_cc,
                                            bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.xgboost_button_cc.grid(row=4, column=2, pady=10, padx=10)
        self.xgboost_output_text_cc = tk.Text(self.output_frame, height=2, width=18)
        self.xgboost_output_text_cc.grid(row=4, column=3, padx=10, pady=10)
        developer_info = tk.Label(text="This GUI is developed by combined efforts of: \n Muhammad Saud Khan (khans28@myumanitoba.ca), University of Manitoba, Canada and Zohaib Mehmood (zoohaibmehmood@gmail.com), COMSATS University Islamabad, Pakistan",
                                  bg="light blue", fg="red", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()
    def create_slider(self, text, min_val, max_val, default_val, row):
        label = tk.Label(self.input_frame, text=text, bg="#FFFFFF", font=("Helvetica", 12, "bold"), fg="darkgreen", anchor="w")
        label.grid(row=row*2, column=0, padx=10, pady=5, sticky="w")

        slider = tk.Scale(self.input_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                          resolution=0.01, length=300, bg="#FFFFFF", troughcolor="#DDDDDD", sliderrelief=tk.RAISED, highlightthickness=0)
        slider.set(default_val)
        slider.grid(row=row*2, column=1, columnspan=3, padx=10, pady=5, sticky="w")
        current_label = tk.Label(self.input_frame, text=f"Current: {default_val:.2f}", bg="#FFFFFF", font=("Helvetica", 10, "bold"), fg="red")
        current_label.grid(row=row*2+1, column=1, padx=10, pady=5, sticky="w")
        min_label = tk.Label(self.input_frame, text=f"Min: {min_val:.2f}", bg="#FFFFFF", font=("Helvetica", 10))
        min_label.grid(row=row*2+1, column=0, padx=10, pady=5, sticky="w")
        max_label = tk.Label(self.input_frame, text=f"Max: {max_val:.2f}", bg="#FFFFFF", font=("Helvetica", 10))
        max_label.grid(row=row*2+1, column=2, padx=10, pady=5, sticky="w")
        def update_current_label(value):
            current_label.config(text=f"Current: {float(value):.2f}")
        slider.config(command=update_current_label)
        setattr(self, f'slider_{row}', slider)
    def calculate_y_c(self):
        d1 = self.slider_1.get()
        d2 = self.slider_3.get()
        d3 = self.slider_5.get()
        d4 = self.slider_7.get()
        d5 = self.slider_9.get()
        d6 = self.slider_11.get()
        if  (d1*d4)+d6 == 0 or d3 == 0 or d1 == 0 or self.G3C85 == 0 or d3+d6 == 0:
            raise ValueError("Check input data")
        y = 0.0
        y += (sqrt(self.G1C65)/(((d6+d1)/(d3+d6))+(self.G1C65+self.G1C75)))
        y *= (((d3-self.G2C45)-self.G2C75)/((d1*d4)+d6))
        y *= ((sqrt(((d5/d1)-(d4-d6)))/self.G3C85)/d3)
        y *= ((((d5+self.G4C95)-self.G4C05)/(self.G4C95+self.G4C65))/((d2+d4)+(d5+self.G4C65)))
        self.gep_output_text_c.delete(1.0, tk.END)
        self.gep_output_text_c.insert(tk.END, f"{y:.6f}")
    def calculate_y_a(self):
        d1 = self.slider_1.get()
        d2 = self.slider_3.get()
        d3 = self.slider_5.get()
        d4 = self.slider_7.get()
        d5 = self.slider_9.get()
        d6 = self.slider_11.get()
        if d6 == 0 or self.G1C2 == 0:
            raise ValueError("Check input data")
        y = 0
        y += (((self.G1C0 * d3) * d1) + (d3 + d6)) - ((self.G1C2 / self.G1C2) * (self.G1C3 + self.G1C4))
        y += (((d1 * d2) * self.G2C3) - (self.G2C3 * d5)) - ((self.G2C3 * self.G2C3) * (self.G2C4 + self.G2C3))
        y += ((self.G3C2 + d4) * (d2 + d5)) - ((d5 - d3) * (self.G3C4 - self.G3C1))
        y += ((((d6 - d3) / d6) * (self.G4C1 * self.G4C1)) + (d3 + (d4 * self.G4C4)))
        self.gep_output_text_a.delete(1.0, tk.END)
        self.gep_output_text_a.insert(tk.END, f"{y:.2f}")
    def calculate_y_b(self):
        d1 = self.slider_1.get()
        d2 = self.slider_3.get()
        d3 = self.slider_5.get()
        d4 = self.slider_7.get()
        d5 = self.slider_9.get()
        d6 = self.slider_11.get()
        if  d2-self.G44C4 == 0 or d1 == 0 or d6 == 0:
            raise ValueError("Check input data")
        y = 0.0
        y += ((sqrt(d2) * d4) - (((self.G11C0 + d2) / (d6 + d4))+(d2 - d1)))
        y += (sqrt((d1+d1))*(d1*sqrt(((d3+d1)-self.G22C7))))
        y += sqrt(((d3-(d6-((d5/d1)/(d6+d6))))*d1))
        y += (d1-sqrt((d4+(((d2/d6)*(d6-d4))/(d2-self.G44C4)))))
        self.gep_output_text_b.delete(1.0, tk.END)
        self.gep_output_text_b.insert(tk.END, f"{y:.2f}")
    def calculate_xgboost_aa(self):
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\software-and-prediction-main"
            filename = r"Density.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            print("Excel file loaded successfully.")
            print(f"Excel file head:\n{df.head()}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=500)
            print("Train-test split done.")
            print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
            print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
            regressor = MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=50,
                reg_lambda=0.01,
                gamma=1,
                max_depth=5
            ))
            model = regressor.fit(x_train, y_train)
            print("Model training completed.")
            new_input = np.array([[self.slider_1.get(), self.slider_3.get(), self.slider_5.get(), self.slider_7.get(), self.slider_9.get(), self.slider_11.get()]])
            print(f"New input for prediction: {new_input}")
            y_pred = model.predict(new_input)
            print(f"Prediction result: {y_pred}")
            self.xgboost_output_text_aa.delete(1.0, tk.END)
            self.xgboost_output_text_aa.insert(tk.END, f"{y_pred[0][0]:.2f}")
        except FileNotFoundError:
            self.xgboost_output_text_aa.delete(1.0, tk.END)
            self.xgboost_output_text_aa.insert(tk.END, "Error: Excel file not found")
            print("Error: Excel file not found")
        except ValueError as ve:
            self.xgboost_output_text_aa.delete(1.0, tk.END)
            self.xgboost_output_text_aa.insert(tk.END, "Error: Invalid data format")
            print("Error:", ve)
        except Exception as e:
            self.xgboost_output_text_aa.delete(1.0, tk.END)
            self.xgboost_output_text_aa.insert(tk.END, "Error: Prediction failed")
            print("Error:", e)
    def calculate_xgboost_bb(self):
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\software-and-prediction-main"
            filename = r"Shrinkage.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            print("Excel file loaded successfully.")
            print(f"Excel file head:\n{df.head()}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=500)
            print("Train-test split done.")
            print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
            print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
            regressor = MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=50,
                reg_lambda=0.01,
                gamma=1,
                max_depth=5
            ))
            model = regressor.fit(x_train, y_train)
            print("Model training completed.")
            new_input = np.array([[self.slider_1.get(), self.slider_3.get(), self.slider_5.get(), self.slider_7.get(), self.slider_9.get(), self.slider_11.get()]])
            print(f"New input for prediction: {new_input}")
            y_pred = model.predict(new_input)
            print(f"Prediction result: {y_pred}")
            self.xgboost_output_text_bb.delete(1.0, tk.END)
            self.xgboost_output_text_bb.insert(tk.END, f"{y_pred[0][0]:.2f}")
        except FileNotFoundError:
            self.xgboost_output_text_bb.delete(1.0, tk.END)
            self.xgboost_output_text_bb.insert(tk.END, "Error: Excel file not found")
            print("Error: Excel file not found")
        except ValueError as ve:
            self.xgboost_output_text_bb.delete(1.0, tk.END)
            self.xgboost_output_text_bb.insert(tk.END, "Error: Invalid data format")
            print("Error:", ve)
        except Exception as e:
            self.xgboost_output_text_bb.delete(1.0, tk.END)
            self.xgboost_output_text_bb.insert(tk.END, "Error: Prediction failed")
            print("Error:", e)
    def calculate_xgboost_cc(self):
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\software-and-prediction-main"
            filename = r"Strain.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=500)
            print("Train-test split done.")
            regressor = MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=50,
                reg_lambda=0.01,
                gamma=1,
                max_depth=5
            ))
            model = regressor.fit(x_train, y_train)
            print("Model training completed.")
            new_input = np.array([[self.slider_1.get(), self.slider_3.get(), self.slider_5.get(), self.slider_7.get(), self.slider_9.get(), self.slider_11.get()]])
            print(f"New input for prediction: {new_input}")
            y_pred = model.predict(new_input)
            print(f"Prediction result: {y_pred}")
            self.xgboost_output_text_cc.delete(1.0, tk.END)
            self.xgboost_output_text_cc.insert(tk.END, f"{y_pred[0][0]:.8f}")
        except FileNotFoundError:
            self.xgboost_output_text_cc.delete(1.0, tk.END)
            self.xgboost_output_text_cc.insert(tk.END, "Error: Excel file not found")
            print("Error: Excel file not found")
        except ValueError as ve:
            self.xgboost_output_text_cc.delete(1.0, tk.END)
            self.xgboost_output_text_cc.insert(tk.END, "Error: Invalid data format")
            print("Error:", ve)
        except Exception as e:
            self.xgboost_output_text_cc.delete(1.0, tk.END)
            self.xgboost_output_text_cc.insert(tk.END, "Error:Prediction failed")
            print("Error:", e)
if __name__ == "__main__":
    root = tk.Tk()
    gui = RangeInputGUI(root)
    root.mainloop()
#Thanks:)  > Engineer Muhammad Saud Khan 