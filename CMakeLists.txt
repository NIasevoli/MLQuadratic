cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(example-app)

# Forza CMake a ignorare CUDA e Kineto nei flag globali
set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(USE_KINETO OFF CACHE BOOL "" FORCE)

# =================================================================
# AGGIRAMENTO ERRORE "LIBRARY KINETO NOT FOUND" SU WINDOWS CPU
# =================================================================
# Creiamo fisicamente un file kineto.lib vuoto nella cartella di build
# per soddisfare il controllo rigido del file system di LibTorch.
set(DUMMY_KINETO_DIR "${CMAKE_CURRENT_BINARY_DIR}/kineto_stub")
file(MAKE_DIRECTORY "${DUMMY_KINETO_DIR}")
file(WRITE "${DUMMY_KINETO_DIR}/kineto.lib" "")

# Aggiungiamo questa cartella temporanea ai percorsi di ricerca di CMake
list(APPEND CMAKE_LIBRARY_PATH "${DUMMY_KINETO_DIR}")
# =================================================================

# Ora find_package passerà senza problemi
find_package(Torch REQUIRED)

# Puliamo TORCH_LIBRARIES per evitare che il compilatore provi davvero 
# a linkare il file kineto.lib vuoto creato sopra.
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "kineto")

set(CMAKE_CXX_STANDARD 20)

# Dichiarazione dell'eseguibile
add_executable(example-app main.cpp)

# Definizioni fondamentali per disattivare il profiler negli header C++
target_compile_definitions(example-app PRIVATE 
    DISABLE_TORCH_KINETO
    USE_CPU
)

# Collegamento delle librerie reali di PyTorch
target_link_libraries(example-app PRIVATE "${TORCH_LIBRARIES}")

# Configurazione dello standard C++ richiesto da PyTorch
target_compile_features(example-app PRIVATE cxx_std_17)

# Copia automatica delle DLL di LibTorch nella cartella dell'eseguibile
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:example-app>
    )
endif()