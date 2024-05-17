 #LIBRARY
import streamlit as st
from PIL import Image
from skimage.data import shepp_logan_phantom
from skimage.transform import  iradon
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate



# Main function
def main():
    st.title('CT Simulator')
    
    # Upload image
    radio_button = st.sidebar.radio("Jenis Citra", ("Citra Homogen", "Citra Semi Homogen", "Citra Kompleks"))
    if radio_button=="Citra Kompleks":
        uploaded = st.sidebar.file_uploader("Unggah citra kompleks", type=["jpg", "jpeg", "png", "tif"])

    translasi = st.sidebar.selectbox("Jumlah translasi", [5, 10, 20, 50, 100, 250, 500])

    derajat_rot = st.sidebar.number_input("Derajat rotasi", min_value=1, max_value=180, step=1)

    #submit button
    submit_button = st.sidebar.button('Submit')

    # Process image if uploaded
    if submit_button and radio_button is not None:
        #image = Image.open(image)
        
        if radio_button=="Citra Kompleks":
            image = Image.open(uploaded)
            #file_name = uploaded.name
            #file_name = list(uploaded.keys())[0]
#            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.convert('L')  # Konversi ke skala abu-abu
            image = np.array(image)#image = cv2.imread(image)
#            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (500, 500))

        elif radio_button=="Citra Semi Homogen":
            # Membuat latar belakang persegi dengan elemen matriks 0
            image = np.zeros((500, 500))

            # Menghitung pusat lingkaran
            center_x = 250
            center_y = 250

            # Menghitung radius lingkaran besar
            radius_big = 200

            # Menghitung panjang sisi persegi kecil untuk "lubang"
            square_size = 50

            # Menghitung batas-batas persegi kecil
            left = center_x
            right = center_x + square_size
            top = center_y + square_size
            bottom = center_y

            # Membuat grid untuk persegi dengan latar belakang 0
            x = np.linspace(0, 499, 500)
            y = np.linspace(0, 499, 500)
            X, Y = np.meshgrid(x, y)

            # Membuat mask untuk bagian lingkaran besar (di mana jarak kurang dari atau sama dengan radius besar)
            distance_big = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            circle_mask_big = distance_big <= radius_big

            # Membuat mask untuk bagian persegi kecil
            square_mask = (X >= left) & (X <= right) & (Y >= bottom) & (Y <= top)

            # Mengganti nilai matriks pada bagian lingkaran besar menjadi 1
            image[circle_mask_big] = 1

            # Mengurangi nilai matriks pada bagian persegi kecil dari lingkaran besar
            image[square_mask] = 0


        elif radio_button=="Citra Homogen":
            # Membuat latar belakang persegi dengan elemen matriks 0
            image = np.zeros((500, 500))

            # Menghitung pusat lingkaran
            center_x = 250
            center_y = 250

            # Menghitung radius lingkaran
            radius = 200

            # Membuat grid untuk persegi dengan latar belakang 0
            x = np.linspace(0, 499, 500)
            y = np.linspace(0, 499, 500)
            X, Y = np.meshgrid(x, y)

            # Menghitung jarak dari setiap titik grid ke pusat lingkaran
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

            # Membuat mask untuk bagian lingkaran (di mana jarak kurang dari atau sama dengan radius)
            circle_mask = distance <= radius
    
            # Mengganti nilai matriks pada bagian lingkaran menjadi 1
            image[circle_mask] = 1



        #inisiasi
        size=image.shape[0]
        rs = np.linspace(-1, 1, image.shape[0])
        thetas = np.arange(0,180,derajat_rot) * np.pi/180
        dtheta = np.diff(thetas)[0]
        dr = np.diff(rs)[0]

        #perhitungan RS 
        rso=np.sum(image, axis=1)
        rotations = np.array([rotate(image, theta*180/np.pi) for theta in thetas])
        rsrot = np.array([rotation.sum(axis=0) * dr for rotation in rotations])

        #mengatur ulang RS sesuai jumlah translasi
        block_sum= size/translasi
        idx = np.arange(0, size+1, block_sum)
        idx_int= np.ceil(idx).astype(int)
        def sum_elements(matrix, indices):
            sums = []
            for row in matrix:
                row_sums = []
                for i in range(len(indices) - 1):
                    start_idx = indices[i]
                    end_idx = indices[i + 1]
                    row_sums.append(np.sum(row[start_idx:end_idx]))
                sums.append(row_sums)
            return np.array(sums)
        rsrot = sum_elements(rsrot, idx_int)

        #1variabel plot sebelum pengurangan RS & ekstensi data
        rsrot_T=rsrot.T
        rsn=np.linspace(-1, 1, rsrot_T.shape[0])
        recons=iradon(rsrot_T, filter_name='ramp')


        # pengurangan jumlah raysum
        def rdc1(x):
            return(rso[::2])

        rso=rdc1(rso)

        def rdc(x):
            return(rsrot[:, ::2])

        rsrot_rdc=rdc(rsrot)

        #2variabel plot setelah pengurangan RS & sebelum ekstensi data
        rsrot_rdc_T=rsrot_rdc.T
        rs_rdc=np.linspace(-1,1, rsrot_rdc.shape[1])
        recons_rdc=iradon(rsrot_rdc_T, filter_name='ramp')

        #FUUNGSI INTERPOLASI
        arr = rsrot_rdc
        def interp(x):
            return (x[:-1] + x[1:]) / 2
        new_data = np.array([interp (row) for row in arr])
        new_rows, new_cols = new_data.shape
        rsrot_interp = np.empty((arr.shape[0], arr.shape[1] + new_cols), dtype=float)
        rsrot_interp[:, ::2] = arr
        rsrot_interp[:, 1::2] = new_data

        #3variabel plot setelah pengurangan RS & setelah ekstensi data
        rsrot_interp_T=rsrot_interp.T
        rint=np.linspace(-1,1,rsrot_interp.shape[1])
        recons_int=iradon(rsrot_interp_T, filter_name='ramp')



        # Display images in 2x3 layout
        col1, col2 = st.columns([5.7, 4.3])

        # Bagi daftar gambar menjadi dua bagian
        images_part1 = image
        fig0, ax0 = plt.subplots(figsize=(8, 6))
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        fig4, ax4 = plt.subplots(figsize=(3, 3))
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        fig6, ax6 = plt.subplots(figsize=(3, 3))


        # Tampilkan gambar-gambar dari bagian pertama di baris pertama
        with col1:

            # Checkbox untuk memilih grafik
#            show_graph1 = st.checkbox('Grafik 1')
#            show_graph2 = st.checkbox('Grafik 2')
#            show_graph3 = st.checkbox('Grafik 3')
            # Plot grafik berdasarkan pilihan checkbox
#            if show_graph1:
               #plt.figure(figsize=(8, 6)) 
#               plt.plot(rsn, rsrot_T[:,0], label='Grafik 1')
#            if show_graph2:
               #plt.figure(figsize=(8, 6)) 
#               plt.plot(rs_rdc, rsrot_rdc_T[:,0], label='Grafik 2')
#            if show_graph3:
               #plt.figure(figsize=(8, 6)) 
#               plt.plot(rint, rsrot_interp_T[:,0], label='Grafik 3')


            st.write ('Plot Profil intensitas')
#            plt.figure(figsize=(8, 6))
            ax0.plot(rsn, rsrot_T[:,0], label='Grafik 1')
            ax0.plot(rs_rdc, rsrot_rdc_T[:,0], label='Grafik 2')
            ax0.plot(rint, rsrot_interp_T[:,0], label='Grafik 3')
            # Menambahkan legenda
            ax0.legend()
            # Menampilkan plot
            st.pyplot(fig0)
            st.write('Grafik1: sebelum pengurangan RS & ekstensi data')
            st.write('Grafik2: setelah pengurangan RS & sebelum ekstensi data')
            st.write('Grafik3: setelah pengurangan RS & setelah ekstensi data')

        with col2:
            st.write ('Citra Asli')
            st.image(images_part1, use_column_width=True)

        # Tampilkan gambar-gambar dari bagian kedua di baris kedua
        with col1:
#            st.write ('Sinogram sebelum pengurangan RS & ekstensi data')
            ax1.pcolor(thetas, rsn, rsrot_T,  cmap='gray')
            ax1.set_xlabel(r'$\theta$', fontsize=20)
            ax1.set_ylabel('$r$', fontsize=20)
        with col2:
#            st.write ('Citra hasil rekonstruksi sebelum pengurangan RS & ekstensi data')
            ax2.imshow(recons, cmap='gray')
            ax2.axis('off')

        # Tampilkan gambar-gambar dari bagian pertama di baris ketiga
        with col1:
#            st.write ('Sinogram setelah pengurangan RS & sebelum ekstensi data')
            ax3.pcolor(thetas, rs_rdc, rsrot_rdc_T,  cmap='gray')
            ax3.set_xlabel(r'$\theta$', fontsize=20)
            ax3.set_ylabel('$r$', fontsize=20)
        with col2:
#            st.write ('Citra hasil rekonstruksi setelah pengurangan RS & sebelum ekstensi data')
            ax4.imshow(recons_rdc, cmap='gray')
            ax4.axis('off')

        # Tampilkan gambar-gambar dari bagian kedua di baris keempat
        with col1:
#            st.write ('Sinogram setelah pengurangan RS & setelah ekstensi data')
            ax5.pcolor(thetas, rint, rsrot_interp_T, shading='auto', cmap='gray')
            ax5.set_xlabel(r'$\theta$', fontsize=20)
            ax5.set_ylabel('$r$', fontsize=20)
        with col2:
#            st.write ('Citra hasil rekonstruksi setelah pengurangan RS & setelah ekstensi data')
            ax6.imshow(recons_int, cmap='gray')
            ax6.axis('off')

        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)
        st.pyplot(fig4)
        st.pyplot(fig5)
        st.pyplot(fig6)

if __name__ == '__main__':
    main()
