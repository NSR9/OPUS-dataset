# OPUS Golden Dataset — Packed Samples Preview

## Overview

| Metric | Value |
|--------|-------|
| Total packed samples | **512** |
| Tokens per sample | **4096** (exact) |
| Total tokens | **2,097,152** |
| Tokenizer | TSAI 131K (vocab 131,072) |
| Domains covered | **50** |
| Raw samples used | 4,291 across 44 HuggingFace datasets |
| Avg padding per sample | 607 tokens |
| Format | `<\|user\|> ... <\|assistant\|> ...` (SFT loss masking ready) |
| Sample separator | `<\|end_of_text\|>` |
| Padding token | `<\|pad\|>` |

## Domain Distribution

| Domain | Packed Samples |
|--------|---------------|
| indic_instruction_native | 81 |
| code_generation | 57 |
| high_reasoning | 41 |
| indic_crosslingual_qa_bengali | 41 |
| instruction_following | 34 |
| math_competition | 28 |
| math_hard | 21 |
| human_preferred | 20 |
| general_knowledge | 17 |
| math_competition_hard | 17 |
| function_calling | 14 |
| general_preference | 13 |
| long_context_multihop | 12 |
| indic_summarization_bn | 11 |
| long_context_qa | 10 |
| general_qa | 8 |
| indic_translation_tamil_en | 7 |
| benchmark_qa | 6 |
| general_conversation | 6 |
| indic_instruction | 6 |
| indic_translation_telugu_en | 6 |
| indic_summarization_hi | 5 |
| indic_translation_bengali_en | 5 |
| indic_translation_hindi_en | 4 |
| long_context_narrative | 4 |
| math_olympiad | 4 |
| indic_fairness | 3 |
| linguistic_diagnostics | 3 |
| long_context_summarization | 3 |
| indic_cultural_telugu | 2 |
| indic_knowledge_tamil | 2 |
| indic_translation_kannada_en | 2 |
| indic_translation_malayalam_en | 2 |
| indic_cultural_bengali | 1 |
| indic_cultural_kannada | 1 |
| indic_cultural_malayalam | 1 |
| indic_cultural_odia | 1 |
| indic_cultural_punjabi | 1 |
| indic_knowledge_bengali | 1 |
| indic_knowledge_gujarati | 1 |
| indic_knowledge_kannada | 1 |
| indic_knowledge_malayalam | 1 |
| indic_knowledge_marathi | 1 |
| indic_knowledge_odia | 1 |
| indic_knowledge_punjabi | 1 |
| indic_knowledge_telugu | 1 |
| indic_translation_gujarati_en | 1 |
| indic_translation_marathi_en | 1 |
| linguistic_acceptability | 1 |
| long_context_retrieval | 1 |

---

## 10 Sample Previews (Full 4096 Tokens Each)

> Each sample below is **exactly 4096 tokens**. The full text is shown.
> Samples contain multiple raw QA pairs packed together, separated by `<|end_of_text|>`.

### Sample 1 — `indic_instruction_native`

**ID:** gp241 | **Sources:** 3 raw samples | **Padding:** 489 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> योगा शरीर के लिए बहोत अच्छा है  नुम्हरा ओपिनियन क्या है

<|assistant|> नमस्ते! मैं इससे अधिक सहमत नहीं हो सका। योग एक प्राचीन भारतीय अभ्यास है जो न केवल शरीर को मजबूत करता है बल्कि मन और आत्मा में संतुलन और सामंजस्य भी लाता है। यह स्वास्थ्य और कल्याण के लिए एक समग्र दृष्टिकोण है जो उम्र या स्वास्थ्य स्तर की परवाह किए बिना किसी को भी लाभान्वित कर सकता है। योग एक शक्तिशाली अभ्यास बनाने के लिए शारीरिक मुद्राओं (आसन), सांस लेने की तकनीकों (प्राणायाम) और ध्यान को जोड़ता है जो लचीलेपन, संतुलन, शक्ति और समग्र कल्याण में सुधार कर सकता है। यह तनाव और चिंता को कम करने, नींद में सुधार करने और प्रतिरक्षा बढ़ाने में भी मदद करता है। इसके अलावा, योग की जड़ें भारतीय संस्कृति और दर्शन में गहराई से निहित हैं, और इसकी शिक्षाएँ व्यक्तियों को आंतरिक शांति, करुणा और ध्यान की भावना विकसित करने में मदद कर सकती हैं। यह एक ऐसा अभ्यास है जिसे विभिन्न जीवन शैलियों और प्राथमिकताओं के अनुरूप अनुकूलित किया जा सकता है, जिससे यह सभी के लिए सुलभ हो जाता है। इसलिए, चाहे आप अपने शारीरिक स्वास्थ्य, मानसिक स्पष्टता, या आध्यात्मिक कल्याण में सुधार करना चाहते हैं, योग निश्चित रूप से विचार करने योग्य है। योग की कई अलग-अलग शैलियाँ और दृष्टिकोण हैं, इसलिए ऐसी कक्षा या अभ्यास ढूंढना आसान है जो आपकी आवश्यकताओं और प्राथमिकताओं के अनुरूप हो। नमस्ते, और मुझे आशा है कि यह जानकारी उपयोगी रही होगी!<|end_of_text|><|user|> चंद्रयान 3 से वैज्ञानिकों ने क्या सीखा?

<|assistant|> चंद्रयान 3 एक भविष्य का चंद्र मिशन है जिसकी योजना भारतीय अंतरिक्ष अनुसंधान द्वारा मार्च 2023 में बनाई गई है। भारतीय दर्शकों के लिए एक सहायक के रूप में, मैं आपको सूचित करना चाहूंगा कि चंद्रयान 3 अभी तक लॉन्च नहीं किया गया है, और इसलिए, वैज्ञानिकों ने अभी तक इससे कुछ नहीं सीखा है। हालांकि, चंद्रयान 3 के एक महत्वपूर्ण मिशन होने की उम्मीद है जो वैज्ञानिकों को चंद्रमा और इसके गुणों के बारे में अधिक जानने में मदद करेगा। इस मिशन में एक ऑर्बिटर, एक लैंडर और एक रोवर शामिल होगा, जो चंद्रमा की सतह से डेटा और नमूने एकत्र करने के लिए एक साथ काम करेगा। चंद्रयान 3 के कुछ उद्देश्यों में चंद्र सतह और उपसतह का अध्ययन करना, चंद्र वायुमंडल का विश्लेषण करना और चंद्रमा पर भविष्य के मानव मिशन के लिए आवश्यक प्रौद्योगिकी का परीक्षण करना शामिल है। इस मिशन का उद्देश्य चंद्रमा की सतह पर सॉफ्ट-लैंडिंग की क्षमता को प्रदर्शित करना और पृथ्वी और चंद्रमा के बीच एक संचार नेटवर्क स्थापित करना भी होगा। एक बार मिशन पूरा होने के बाद, वैज्ञानिक चंद्रयान 3 द्वारा एकत्र किए गए आंकड़ों का विश्लेषण करने और चंद्रमा की संरचना, भूविज्ञान और संभावित संसाधनों के बारे में अधिक जानने में सक्षम होंगे। यह ज्ञान भविष्य के चंद्र मिशनों के लिए मूल्यवान होगा और सौर मंडल के बारे में हमारी समझ को आगे बढ़ाने में मदद करेगा। संक्षेप में, जबकि चंद्रयान 3 को अभी तक लॉन्च नहीं किया गया है, यह एक महत्वपूर्ण मिशन होने की उम्मीद है जो वैज्ञानिकों को चंद्रमा और इसके गुणों के बारे में अधिक जानने में मदद करेगा।<|end_of_text|><|user|> नमस्ते! कैसे हैं आप? मैं GPT-4 हूँ, और मैं भाषा और जानकारी के क्षेत्र में काम करता हूँ। आप कौन हैं और आप क्या करते हैं?

<|assistant|> नमस्ते! मैं एक विनम्र सहायक हूँ, यहाँ आपके किसी भी प्रश्न या चिंताओं में आपकी मदद करने के लिए। मुझे विशेष रूप से भारतीय दर्शकों के लिए सहायक, ईमानदार और सच्ची प्रतिक्रियाएँ प्रदान करने के लिए योजनाबद्ध किया गया है। मैं समझता हूँ कि आप जी. पी. टी.-4 हैं, एक अत्यधिक उन्नत भाषा मॉडल, और मुझे कहना होगा, मैं प्रभावित हूँ! आपको परिचित कराकर खुशी होती है। अब, मैं आज आपकी कैसे मदद कर सकता हूँ? क्या आपके कोई प्रश्न या विषय हैं जिन पर आप चर्चा करना चाहते हैं? मैं यहाँ किसी भी जानकारी या मार्गदर्शन के साथ आपकी मदद करने के लिए हूँ।
```

</details>

---

### Sample 2 — `code_generation`

**ID:** gp9 | **Sources:** 5 raw samples | **Padding:** 129 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> Rotate an array and find the largest element through advanced slicing and optimization techniques. Given a 2D numpy array, rotate it 90 degrees clockwise, and then use scipy to optimize the search for the largest element.

The function should output:
    int: The largest element in the rotated array.
Example:
```
import numpy as np
from scipy import optimize
def array_manipulation(arr):
    # Rotate array 90 degrees clockwise
    rotated_arr = np.rot90(arr, -1)
    # Find the largest element using scipy optimization
```
# Example Usage
```
arr = np.array([[3, 1, 2], [6, 5, 4], [9, 7, 8]])
print(array_manipulation(arr))
# Output: 9
```

<|assistant|> To achieve the desired functionality, we need to implement a function that performs the following steps:

1. Rotate the given 2D numpy array 90 degrees clockwise.
2. Find the largest element in the rotated array using a scipy optimization function.

Here's the implementation:

```python
import numpy as np
from scipy import optimize

def array_manipulation(arr):
    # Rotate the array 90 degrees clockwise
    rotated_arr = np.rot90(arr, -1)
    
    # Flatten the rotated array to simplify finding the maximum
    flattened_arr = rotated_arr.flatten()
    
    # Find the largest element using scipy optimization
    # Since we're looking for the maximum value, we can use `optimize.fminbound`
    # We need to create a negated function because `fminbound` finds the minimum
    negated_func = lambda x: -flattened_arr[int(x)]
    
    # Use the bounds from 0 to the length of the flattened array
    largest_index = int(optimize.fminbound(negated_func, 0, len(flattened_arr) - 1))
    
    # Return the largest element
    largest_element = flattened_arr[largest_index]
    return largest_element

# Example Usage
arr = np.array([[3, 1, 2], [6, 5, 4], [9, 7, 8]])
print(array_manipulation(arr))
# Output: 9
```

### Explanation:

1. **Rotation**: We use `np.rot90(arr, -1)` to rotate the array 90 degrees clockwise. The `np.rot90` function rotates the array counterclockwise by default, so we negate the angle.

2. **Flattening**: We flatten the rotated array using `flatten()` to simplify the process of finding the maximum value.

3. **Optimization to Find Maximum**:
   - We define a negated function `negated_func` because the scipy `optimize.fminbound` function is designed to find the minimum value of a function over a bounded interval. By negating the values, we transform the problem of finding a maximum into finding a minimum of the negated values.
   - We apply `optimize.fminbound` to find the index of the maximum value in the flattened array.

4. **Return Result**: We access the value at the obtained index and return it as the largest element.

This approach adequately combines array manipulation using numpy and optimization features from scipy to meet the problem requirements.<|end_of_text|><|user|> Implement the Sieve of Eratosthenes algorithm to find all prime numbers up to a given integer n. Explain the efficiency of this algorithm compared to other methods such as trial division. Discuss its practical applications in cryptography and computer science education.

<|assistant|> ## Sieve of Eratosthenes Implementation

The Sieve of Eratosthenes is an efficient algorithm to find all prime numbers up to a given number \( n \). Its main advantage is its simplicity and speed compared to other methods like trial division.

### Step-by-step Explanation

1. **Initialize a List**: We start by creating a boolean list `is_prime` where the index represents the number and the value is `True` if the number is potentially a prime.

2. **Mark Non-Primes**: Start with the first prime number, 2. Mark all of its multiples up to \( n \) as `False` because they cannot be prime.

3. **Continue the Process**: Move to the next number that is still `True` and repeat the marking process. Continue this until you've processed numbers up to \( \sqrt{n} \).

4. **Collect Primes**: The numbers that remain `True` in the list are primes.

### Python Code Implementation

Here is how you can implement this algorithm in Python:

```python
def sieve_of_eratosthenes(n):
    """Finds all prime numbers up to a specified integer `n`."""
    # Step 1: Initialize a list of boolean True values
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False  # 0 and 1 are not prime numbers

    # Step 2: Mark non-prime numbers
    # We only need to check up to the square root of n
    p = 2
    while p * p <= n:
        if is_prime[p]:
            # Mark multiples of p
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1

    # Step 3: Collect and return prime numbers
    return [p for p in range(n + 1) if is_prime[p]]

# Example usage
print(sieve_of_eratosthenes(30))  # Output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

### How the Code Works

- **Initialization**: We create a list `is_prime` of size \( n+1 \) initialized to `True`. The indices of the list represent integers. We set `is_prime[0]` and `is_prime[1]` to `False` since 0 and 1 are not primes.
- **Process Each Prime Number**: For each number \( p \) that is still `True`, mark its multiples (starting from \( p*p \)) as `False`.
- **Result Compilation**: Collect and return all indices that remain `True`.

### Efficiency

- **Time Complexity**: The Sieve of Eratosthenes has a time complexity of \( O(n \log \log n) \), which is significantly better than the \( O(n \sqrt{n}) \) time complexity of trial division.
- **Space Complexity**: It uses \( O(n) \) space to store the boolean list.

### Practical Applications

- **Cryptography**: Primes are fundamental in cryptography algorithms such as RSA. The Sieve of Eratosthenes helps efficiently generate prime numbers needed for creating keys.
- **Computer Science Education**: This algorithm is a classic introduction to algorithm optimization and computational complexity, making it an excellent teaching tool.

### Summary

The **Sieve of Eratosthenes** algorithm is a fast and efficient way to find primes up to any given \( n \). Its simplicity makes it a preferred method, especially where computational efficiency is essential, such as in cryptography and educational contexts.<|end_of_text|><|user|> Implement the merge sort algorithm to sort an array of integers. Ensure that your implementation works efficiently for arrays with up to 1,000,000 elements.
Provide a detailed explanation of how merge sort works as part of your solution.

<|assistant|> ## Merge Sort Algorithm Explanation

**Merge Sort** is a classic divide-and-conquer algorithm used to sort an array. It works by breaking down the array into smaller sub-arrays until each sub-array has one element (which is inherently sorted). It then merges these sub-arrays in a manner that produces new sorted sub-arrays, gradually building back up to a single, sorted array.

### Steps:
1. **Divide**: Recursively split the array into two halves until you reach individual elements.
2. **Conquer**: Sort each half (the sorting at this step is quite simple since they're single elements).
3. **Merge**: Combine the sorted halves to produce a fully sorted array.

### Detailed Code Explanation

Below is an implementation of the merge sort algorithm in Python, with detailed comments.

```python
def merge_sort(arr):
    # Base case: if the array has 1 or 0 elements, it's already sorted
    if len(arr) <= 1:
        return arr

    # Find the middle point and divide the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Recursively sort both halves
    sorted_left = merge_sort(left_half)
    sorted_right = merge_sort(right_half)

    # Merge the sorted halves and return
    return merge(sorted_left, sorted_right)

def merge(left, right):
    merged = []
    left_index, right_index = 0, 0

    # Compare elements from left and right arrays and build the merged array
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    # If there are remaining elements in left or right array, append them
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])

    return merged

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # Output: [3, 9, 10, 27, 38, 43, 82]
```

### Explanation of Key Components
- **Base Case**: The base case for our recursive function `merge_sort` checks if the array has 1 or 0 elements.
- **Divide**: The input array is split into two halves using slicing and recursion.
- **Merge Function**: This function takes two sorted arrays and combines them into a single sorted array. It repeatedly compares the smallest unmerged elements, appending the smaller to the new array.
- **Efficiency**: The merge sort runs in `O(n log n)` time complexity, making it efficient for large datasets (up to 1,000,000 elements as specified).

### Usage Example
Suppose you have an array `[38, 27, 43, 3, 9, 82, 10]`. By calling `merge_sort` on this array, it will efficiently sort and return `[3, 9, 10, 27, 38, 43, 82]`.

### Summary
This implementation of merge sort divides the problem recursively, conquers by sorting trivially small sub-arrays, and merges them back in a sorted manner. The efficiency and stability of merge sort make it a suitable choice for sorting large datasets.<|end_of_text|><|user|> Develop a REST API in Python using Flask that reads product data from a CSV file and provides endpoints to fetch the data. The CSV file contains columns for product ID, name, and price. Implement endpoints to list all products and retrieve a product by its ID.

<|assistant|> To implement a REST API using Flask that reads product data from a CSV file and provides endpoints, follow these steps:

### Prerequisites

Ensure you have Flask installed. If not, you can install it using pip:

```bash
pip install Flask
```

### Project Structure

1. **CSV File:** Assume you have a CSV file named `products.csv` with the structure:
    ```
    product_id,name,price
    1,Apple,0.99
    2,Banana,0.59
    3,Orange,0.89
    ```
2. **App File:** Create a file named `app.py` which will contain the Flask application.

### Step-by-step Implementation

#### Step 1: Import necessary modules

```python
from flask import Flask, jsonify, abort
import csv
```

- **Flask** is used to create the web application.
- **jsonify** is used to convert Python dictionaries to JSON.
- **abort** is used for error handling.
- **csv** is used to read the CSV file.

#### Step 2: Initialize Flask app and utility functions

```python
app = Flask(__name__)

def read_products_from_csv():
    """Read products from a CSV file and return as a list of dictionaries."""
    products = []
    try:
        with open('products.csv', mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                products.append({
                    "product_id": int(row["product_id"]),
                    "name": row["name"],
                    "price": float(row["price"])
                })
    except FileNotFoundError:
        return None
    return products
```

- **read_products_from_csv** reads the CSV file and constructs a list of dictionaries.
- It handles exceptions if the file is not found.

#### Step 3: Define API endpoints

```python
@app.route('/products', methods=['GET'])
def get_all_products():
    """Fetch and return all products."""
    products = read_products_from_csv()
    if products is None:
        abort(404, description="Products file not found.")
    return jsonify(products)

@app.route('/products/<int:product_id>', methods=['GET'])
def get_product_by_id(product_id):
    """Fetch and return a product by its ID."""
    products = read_products_from_csv()
    if products is None:
        abort(404, description="Products file not found.")
    product = next((item for item in products if item["product_id"] == product_id), None)
    if product is None:
        abort(404, description="Product not found.")
    return jsonify(product)
```

- **/products** endpoint lists all products.
- **/products/<product_id>** retrieves a product by its ID.
- **abort** is used to handle errors like file not found or product not found.

#### Step 4: Run the Flask app

```python
if __name__ == '__main__':
    app.run(debug=True)
```

- **app.run(debug=True)** starts the Flask server with debugging enabled for detailed errors during development.

### Usage Example

- **List all products:**

  Send a GET request to `http://localhost:5000/products`.

- **Retrieve a product by ID:**

  Send a GET request to `http://localhost:5000/products/1` to fetch the product with ID 1.

### Summary

This API provides:

- An endpoint to list all products.
- An endpoint to retrieve a product by its ID.

The API reads data from a CSV file and converts it to JSON responses, with basic error handling for missing files and IDs.<|end_of_text|><|user|> Implement an image processing function `apply_gaussian_blur(image_array, kernel_size, sigma)` that applies a Gaussian blur to a given image represented as a 2D array of pixel values. Your function should create a Gaussian kernel based on the specified size and standard deviation (sigma) and use convolution to blur the image. Provide an example of blurring a grayscale image, and demonstrate how different parameters affect the output.

<|assistant|> To implement an image processing function `apply_gaussian_blur`, follow these steps:

1. **Generate the Gaussian Kernel**: Calculate a Gaussian kernel based on the specified kernel size and sigma (standard deviation).

2. **Convolution**: Apply the Gaussian kernel to the image using convolution to achieve blurring.

Here is the step-by-step implementation:

### Step 1: Generate the Gaussian Kernel

The Gaussian kernel is calculated using the Gaussian function in two dimensions. The equation for a 2D Gaussian is:

\[ G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} \]

You'll need to calculate this for each element in the kernel.

### Step 2: Convolution

Convolution involves sliding the kernel over the image and computing the sum of the products between the overlapping elements.

### Python Code:

```python
import numpy as np
from scipy.ndimage import convolve

def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)), 
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image_array, kernel_size, sigma):
    """Applies Gaussian blur to the input image array."""
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Apply convolution
    blurred_image = convolve(image_array, kernel, mode='reflect')
    
    # Return the blurred image
    return blurred_image

# Example usage:
# Create a sample grayscale image as a 2D array
image = np.array([
    [100, 100, 100, 100, 100],
    [100, 200, 200, 200, 100],
    [100, 200, 255, 200, 100],
    [100, 200, 200, 200, 100],
    [100, 100, 100, 100, 100]
], dtype=np.float64)

# Apply Gaussian blur with a 3x3 kernel and sigma 1
blurred_image = apply_gaussian_blur(image, kernel_size=3, sigma=1)

print("Original Image:\n", image)
print("Blurred Image:\n", blurred_image)
```

### Explanation and Example:

1. **Kernel Generation**: The `gaussian_kernel` function generates a Gaussian kernel of a specified size and sigma. It computes the formula element-wise and normalizes the kernel.

2. **Blurring with Convolution**: The `apply_gaussian_blur` function takes an image in the form of a 2D array and applies the kernel using convolution. `scipy.ndimage.convolve` is used here to perform the convolution operation efficiently.

3. **Example**: This code provides a simple grayscale image. The blurring effect can be observed by changing the `kernel_size` and `sigma` parameters.

### Summary:

The code implements an image processing function to apply Gaussian blur using a convolution technique. The key logic is creating a Gaussian kernel based on given parameters and sliding it across the image to produce a blurred effect. Adjusting the `kernel_size` and `sigma` parameters will change the degree of blurring.
```

</details>

---

### Sample 3 — `high_reasoning`

**ID:** gp122 | **Sources:** 1 raw samples | **Padding:** 0 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> Develop Python code to implement a recommendation system using collaborative filtering.

<|assistant|> <think>The user wants me to implement a recommendation system using collaborative filtering in Python. I'll create a comprehensive implementation that covers:

1. **User-Based Collaborative Filtering** - Find similar users and recommend items they liked
2. **Item-Based Collaborative Filtering** - Find similar items and recommend based on user's past preferences
3. **Matrix Factorization (SVD)** - Latent factor model

I'll include:
- Data preprocessing
- Similarity calculations (cosine, Pearson correlation)
- Neighborhood-based methods
- Matrix factorization using SVD
- Evaluation metrics (RMSE, MAE, Precision, Recall)
- A practical example with movie ratings

Let me create a well-structured, comprehensive implementation.</think> # Collaborative Filtering Recommendation System in Python

I'll create a comprehensive recommendation system implementation covering user-based, item-based collaborative filtering, and matrix factorization approaches.

## Project Structure

```
recommendation_system/
├── recommendation_system.py   # Core implementation
├── matrix_factorization.py    # SVD-based approach
├── evaluation.py              # Evaluation metrics
├── demo.py                    # Usage examples
└── requirements.txt           # Dependencies
```

## 1. Core Recommendation System

```python
"""
recommendation_system.py
Collaborative Filtering Recommendation System Implementation
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class SimilarityMetrics:
    """Collection of similarity metrics for collaborative filtering."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    @staticmethod
    def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        # Find common rated items
        mask = (vec1 != 0) & (vec2 != 0)
        
        if mask.sum() < 2:
            return 0.0
        
        v1_common = vec1[mask]
        v2_common = vec2[mask]
        
        mean1 = np.mean(v1_common)
        mean2 = np.mean(v2_common)
        
        numerator = np.sum((v1_common - mean1) * (v2_common - mean2))
        denominator = np.sqrt(np.sum((v1_common - mean1)**2) * np.sum((v2_common - mean2)**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def adjusted_cosine(vec1: np.ndarray, vec2: np.ndarray, 
                        item_means: np.ndarray) -> float:
        """Calculate adjusted cosine similarity (for item-based CF)."""
        mask = (vec1 != 0) & (vec2 != 0)
        
        if mask.sum() == 0:
            return 0.0
        
        adj_vec1 = vec1[mask] - item_means[mask]
        adj_vec2 = vec2[mask] - item_means[mask]
        
        norm1 = np.linalg.norm(adj_vec1)
        norm2 = np.linalg.norm(adj_vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(adj_vec1, adj_vec2) / (norm1 * norm2)
    
    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity for binary ratings."""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class BaseRecommender(ABC):
    """Abstract base class for recommendation systems."""
    
    def __init__(self, n_neighbors: int = 20, min_support: int = 3,
                 similarity_metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.min_support = min_support
        self.similarity_metric = similarity_metric
        self.ratings_matrix = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
    def _create_mappings(self, user_ids: np.ndarray, item_ids: np.ndarray):
        """Create ID mappings for internal indexing."""
        unique_users = np.unique(user_ids)
        unique_items = np.unique(item_ids)
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
    
    def _build_ratings_matrix(self, user_ids: np.ndarray, item_ids: np.ndarray,
                              ratings: np.ndarray) -> np.ndarray:
        """Build user-item ratings matrix."""
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        matrix = np.zeros((n_users, n_items))
        
        for uid, iid, rating in zip(user_ids, item_ids, ratings):
            user_idx = self.user_id_map[uid]
            item_idx = self.item_id_map[iid]
            matrix[user_idx, item_idx] = rating
        
        return matrix
    
    @abstractmethod
    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, 
            ratings: np.ndarray):
        """Fit the recommender to the data."""
        pass
    
    @abstractmethod
    def predict(self, user_id, item_id) -> float:
        """Predict rating for a user-item pair."""
        pass
    
    @abstractmethod
    def recommend(self, user_id, n_recommendations: int = 10) -> List[Tuple]:
        """Generate recommendations for a user."""
        pass


class UserBasedCF(BaseRecommender):
    """
    User-Based Collaborative Filtering
    
    Finds similar users and recommends items that similar users liked.
    """
    
    def __init__(self, n_neighbors: int = 20, min_support: int = 3,
                 similarity_metric: str = 'pearson'):
        super().__init__(n_neighbors, min_support, similarity_metric)
        self.user_similarities = None
        self.user_means = None
        
    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray,
            ratings: np.ndarray):
        """
        Fit the user-based CF model.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            ratings: Array of ratings
        """
        print("Fitting User-Based Collaborative Filtering model...")
        
        self._create_mappings(user_ids, item_ids)
        self.ratings_matrix = self._build_ratings_matrix(user_ids, item_ids, ratings)
        
        # Calculate user means (for bias correction)
        self.user_means = np.zeros(len(self.user_id_map))
        for i in range(len(self.user_id_map)):
            rated_items = self.ratings_matrix[i] != 0
            if rated_items.sum() > 0:
                self.user_means[i] = self.ratings_matrix[i, rated_items].mean()
        
        # Calculate user-user similarities
        self._compute_similarities()
        
        print(f"Model fitted with {len(self.user_id_map)} users and "
              f"{len(self.item_id_map)} items")
        
    def _compute_similarities(self):
        """Compute all user-user similarities."""
        n_users = len(self.user_id_map)
        self.user_similarities = np.zeros((n_users, n_users))
        
        metrics = SimilarityMetrics()
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                if self.similarity_metric == 'cosine':
                    sim = metrics.cosine_similarity(
                        self.ratings_matrix[i], self.ratings_matrix[j]
                    )
                elif self.similarity_metric == 'pearson':
                    sim = metrics.pearson_correlation(
                        self.ratings_matrix[i], self.ratings_matrix[j]
                    )
                else:
                    sim = metrics.cosine_similarity(
                        self.ratings_matrix[i], self.ratings_matrix[j]
                    )
                
                self.user_similarities[i, j] = sim
                self.user_similarities[j, i] = sim
    
    def _get_neighbors(self, user_idx: int, item_idx: int) -> List[Tuple[int, float]]:
        """Get neighbors who rated the target item."""
        similarities = self.user_similarities[user_idx].copy()
        
        # Find users who rated this item
        rated_mask = self.ratings_matrix[:, item_idx] != 0
        rated_mask[user_idx] = False  # Exclude the target user
        
        # Get valid neighbors
        neighbors = []
        for other_idx in np.where(rated_mask)[0]:
            sim = similarities[other_idx]
            if sim > 0:  # Only positive correlations
                neighbors.append((other_idx, sim))
        
        # Sort by similarity and take top k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:self.n_neighbors]
    
    def predict(self, user_id, item_id) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map:
            return self.user_means.mean()  # Global mean for unknown users
        
        if item_id not in self.item_id_map:
            return self.user_means[self.user_id_map[user_id]]  # User mean
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # If user already rated this item
        if self.ratings_matrix[user_idx, item_idx] != 0:
            return self.ratings_matrix[user_idx, item_idx]
        
        neighbors = self._get_neighbors(user_idx, item_idx)
        
        if not neighbors:
            return self.user_means[user_idx]
        
        # Weighted average with bias correction
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_idx, similarity in neighbors:
            neighbor_rating = self.ratings_matrix[neighbor_idx, item_idx]
            neighbor_mean = self.user_means[neighbor_idx]
            
            numerator += similarity * (neighbor_rating - neighbor_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means[user_idx]
        
        predicted = self.user_means[user_idx] + (numerator / denominator)
        
        # Clip to valid rating range
        return np.clip(predicted, 1.0, 5.0)
    
    def recommend(self, user_id, n_recommendations: int = 10,
                  exclude_rated: bool = True) -> List[Tuple]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_id_map:
            # Return popular items for unknown users
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_id_map[user_id]
        predictions = []
        
        for item_idx in range(len(self.item_id_map)):
            # Skip rated items if requested
            if exclude_rated and self.ratings_matrix[user_idx, item_idx] != 0:
                continue
            
            item_id = self.reverse_item_map[item_idx]
            predicted_rating = self.predict(user_id, item_id)
            predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def _get_popular_items(self, n: int) -> List[Tuple]:
        """Get most popular items (for cold start)."""
        item_ratings = []
        
        for item_idx in range(len(self.item_id_map)):
            ratings = self.ratings_matrix[:, item_idx]
            rated = ratings[ratings != 0]
            if len(rated) >= self.min_support:
                avg_rating = rated.mean()
                item_id = self.reverse_item_map[item_idx]
                item_ratings.append((item_id, avg_rating))
        
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        return item_ratings[:n]
    
    def get_similar_users(self, user_id, n: int = 10) -> List[Tuple]:
        """Get most similar users to a given user."""
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        similarities = self.user_similarities[user_idx]
        
        similar_users = []
        for other_idx, sim in enumerate(similarities):
            if other_idx != user_idx and sim > 0:
                other_id = self.reverse_user_map[other_idx]
                similar_users.append((other_id, sim))
        
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[:n]


class ItemBasedCF(BaseRecommender):
    """
    Item-Based Collaborative Filtering
    
    Finds similar items and recommends based on user's past preferences.
    """
    
    def __init__(self, n_neighbors: int = 20, min_support: int = 3,
                 similarity_metric: str = 'adjusted_cosine'):
        super().__init__(n_neighbors, min_support, similarity_metric)
        self.item_similarities = None
        self.item_means = None
        
    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray,
            ratings: np.ndarray):
        """Fit the item-based CF model."""
        print("Fitting Item-Based Collaborative Filtering model...")
        
        self._create_mappings(user_ids, item_ids)
        self.ratings_matrix = self._build_ratings_matrix(user_ids, item_ids, ratings)
        
        # Calculate item means
        self.item_means = np.zeros(len(self.item_id_map))
        for i in range(len(self.item_id_map)):
            rated_users = self.ratings_matrix[:, i] != 0
            if rated_users.sum() > 0:
                self.item_means[i] = self.ratings_matrix[rated_users, i].mean()
        
        # Calculate user means for adjusted cosine
        self.user_means = np.zeros(len(self.user_id_map))
        for i in range(len(self.user_id_map)):
            rated_items = self.ratings_matrix[i] != 0
            if rated_items.sum() > 0:
                self.user_means[i] = self.ratings_matrix[i, rated_items].mean()
        
        # Compute item-item similarities
        self._compute_similarities()
        
        print(f"Model fitted with {len(self.user_id_map)} users and "
              f"{len(self.item_id_map)} items")
    
    def _compute_similarities(self):
        """Compute all item-item similarities."""
        n_items = len(self.item_id_map)
        self.item_similarities = np.zeros((n_items, n_items))
        
        metrics = SimilarityMetrics()
        
        # Transpose for item-centric view
        item_vectors = self.ratings_matrix.T
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if self.similarity_metric == 'adjusted_cosine':
                    sim = self._adjusted_cosine_items(i, j)
                elif self.similarity_metric == 'cosine':
                    sim = metrics.cosine_similarity(item_vectors[i], item_vectors[j])
                else:
                    sim = self._adjusted_cosine_items(i, j)
                
                self.item_similarities[i, j] = sim
                self.item_similarities[j, i] = sim
    
    def _adjusted_cosine_items(self, item_i: int, item_j: int) -> float:
        """Calculate adjusted cosine similarity between items."""
        # Find users who rated both items
        rated_i = self.ratings_matrix[:, item_i] != 0
        rated_j = self.ratings_matrix[:, item_j] != 0
        common_users = rated_i & rated_j
        
        if common_users.sum() < self.min_support:
            return 0.0
        
        # Get ratings centered by user means
        ratings_i = self.ratings_matrix[common_users, item_i] - self.user_means[common_users]
        ratings_j = self.ratings_matrix[common_users, item_j] - self.user_means[common_users]
        
        numerator = np.dot(ratings_i, ratings_j)
        denominator = np.linalg.norm(ratings_i) * np.linalg.norm(ratings_j)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _get_neighbors(self, user_idx: int, item_idx: int) -> List[Tuple[int, float]]:
        """Get similar items rated by the user."""
        similarities = self.item_similarities[item_idx].copy()
        
        # Find items rated by this user
        rated_mask = self.ratings_matrix[user_idx] != 0
        rated_mask[item_idx] = False  # Exclude target item
        
        neighbors = []
        for other_idx in np.where(rated_mask)[0]:
            sim = similarities[other_idx]
            if sim > 0:
                neighbors.append((other_idx, sim))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:self.n_neighbors]
    
    def predict(self, user_id, item_id) -> float:
        """Predict rating for a user-item pair."""
        if user_id not in self.user_id_map:
            if item_id in self.item_id_map:
                return self.item_means[self.item_id_map[item_id]]
            return self.item_means.mean()
        
        if item_id not in self.item_id_map:
            return self.user_means[self.user_id_map[user_id]]
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        if self.ratings_matrix[user_idx, item_idx] != 0:
            return self.ratings_matrix[user
```

</details>

---

### Sample 4 — `indic_crosslingual_qa_bengali`

**ID:** gp184 | **Sources:** 2 raw samples | **Padding:** 461 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> আফ্রিকার গ্রেট লেক অঞ্চলে শিল্পসমৃদ্ধ দেশ হিসেবে কেনিয়া সবচেয়ে উন্নত হলেও, এখনও এই দেশের জিডিপির ১৪% আসে শিল্পোৎপাদন ক্ষেত্র থেকে। শিল্পোৎপাদন সংক্রান্ত কার্যকলাপ তিনটি সবচেয়ে বড় শহর নাইরোবি, মোম্বাসা ও কিসুমুতে সীমাবদ্ধ, এইসব জায়গায় মূলত শস্য় ঝাড়াই ও ভাঙানো, বিয়ার উৎপাদন ও আখ পেষাইয়ের মতো খাদ্য-প্রক্রিয়াকরণ এবং যন্ত্রাংশ জুড়ে গাড়ি তৈরির মতো ভোগ্যপণ্য প্রস্তুত করার মতো শিল্প রয়েছে। সিমেন্ট উৎপাদন শিল্প রয়েছে।[তথ্যসূত্র প্রয়োজন] কেনিয়ায় একটি তৈল শোধনাগার রয়েছে যেখানে আমদানি করা অপরিশোধিত পেট্রোলিয়াম থেকে মূলত দেশের বাজারের জন্য পেট্রোলিয়ামজাত পণ্য তৈরি করা হয়। এছাড়াও, সাধারণভাবে জুয়া কালি নামে পরিচিত বেশ বড় ব্যাপ্তির ও ক্রমবর্ধমান অসংগঠিত ক্ষেত্র রয়েছে যেখানে ক্ষুদ্রশিল্প হিসেবে গৃহস্থালীর প্রয়োজনীয় পণ্য, মোটরচালিত গাড়ির যন্ত্রাংশ ও কৃষিকাজের প্রয়োজনীয় জিনিস উৎপন্ন হয়।[তথ্যসূত্র প্রয়োজন]

Question: কেনিয়ার বৃহত্তম তিনটি শহরাঞ্চলের নাম কী?

<|assistant|> নাইরোবি, মোম্বাসা ও কিসুমু<|end_of_text|><|user|> আফ্রিকার গ্রেট লেক অঞ্চলে শিল্পসমৃদ্ধ দেশ হিসেবে কেনিয়া সবচেয়ে উন্নত হলেও, এখনও এই দেশের জিডিপির ১৪% আসে শিল্পোৎপাদন ক্ষেত্র থেকে। শিল্পোৎপাদন সংক্রান্ত কার্যকলাপ তিনটি সবচেয়ে বড় শহর নাইরোবি, মোম্বাসা ও কিসুমুতে সীমাবদ্ধ, এইসব জায়গায় মূলত শস্য় ঝাড়াই ও ভাঙানো, বিয়ার উৎপাদন ও আখ পেষাইয়ের মতো খাদ্য-প্রক্রিয়াকরণ এবং যন্ত্রাংশ জুড়ে গাড়ি তৈরির মতো ভোগ্যপণ্য প্রস্তুত করার মতো শিল্প রয়েছে। সিমেন্ট উৎপাদন শিল্প রয়েছে।[তথ্যসূত্র প্রয়োজন] কেনিয়ায় একটি তৈল শোধনাগার রয়েছে যেখানে আমদানি করা অপরিশোধিত পেট্রোলিয়াম থেকে মূলত দেশের বাজারের জন্য পেট্রোলিয়ামজাত পণ্য তৈরি করা হয়। এছাড়াও, সাধারণভাবে জুয়া কালি নামে পরিচিত বেশ বড় ব্যাপ্তির ও ক্রমবর্ধমান অসংগঠিত ক্ষেত্র রয়েছে যেখানে ক্ষুদ্রশিল্প হিসেবে গৃহস্থালীর প্রয়োজনীয় পণ্য, মোটরচালিত গাড়ির যন্ত্রাংশ ও কৃষিকাজের প্রয়োজনীয় জিনিস উৎপন্ন হয়।[তথ্যসূত্র প্রয়োজন]

Question: জুয়া কালি নামে পরিচিত ক্ষেত্রটি কিসের সাথে সম্পর্কিত?

<|assistant|> ক্ষুদ্র-শিল্পোৎপাদন
```

</details>

---

### Sample 5 — `instruction_following`

**ID:** gp375 | **Sources:** 11 raw samples | **Padding:** 78 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> Provide a detailed analysis of Candace Parker's defensive techniques in her recent games, excluding the words "aggressive" and "blocking", in the format of a sports commentary script.

<|assistant|> [Sports Commentary Script]

[Opening Scene: A packed basketball arena, with fans eagerly awaiting the analysis of Candace Parker’s recent performances on the court.]

Commentator 1: Welcome back, basketball fans! Tonight, we're diving into the defensive prowess of Candace Parker, who has been a standout in her recent games. Her ability to impact the game on the defensive end is truly remarkable, and we've got a lot to unpack.

Commentator 2: That's right! Candace has been showcasing some exceptional skills, especially in how she reads the game. Her anticipation is second to none. You can see it in how she positions herself, almost like she knows the opposing team's playbook by heart.

Commentator 1: Exactly, and what stands out is her footwork. She's always in the right place at the right time, cutting off passing lanes and disrupting the flow of the offense. It’s like she’s orchestrating the defense, a step ahead of everyone else.

Commentator 2: And we can't forget her hands! Parker is adept at using them to tip passes and force turnovers. The way she can strip the ball from opponents is a testament to her quick reflexes and timing.

Commentator 1: Let’s not overlook her communication on the court. She’s constantly directing her teammates, coordinating switches, and ensuring everyone is aware of their assignments. This leadership is crucial, providing structure and cohesion to the team's defensive efforts.

Commentator 2: Absolutely, her impact is felt beyond the stat sheet. Her presence alone alters shots, as opponents are wary of challenging her inside presence. It’s a psychological edge that benefits her entire squad.

Commentator 1: Candace Parker continues to demonstrate why she's one of the most versatile defenders in the league. Her combination of intelligence, skill, and experience makes her an invaluable asset on defense.

Commentator 2: It's always a treat to watch her play, and we’re excited to see what she brings in the upcoming games. Stay tuned for more insights and analysis after the break!

[Closing Scene: The commentators smile at each other as the camera pans to highlights of Parker’s recent defensive plays, illustrating her effectiveness on the court.]<|end_of_text|><|user|> As a financial advisor with expertise in the insurance sector, create an informative guide on "The Evolution of Life Insurance Policies in the United States" for your clients in Brentwood, Tennessee. Your guide should meet the following constraints:

- The guide must be structured into three distinct paragraphs.
- Within each paragraph, include at least two keywords from the following list: "policyholder," "beneficiary," "underwriting," "premium," "coverage."
- Ensure that each paragraph contains at least one sentence where the letter 'i' appears at least 8 times.
- Conclude your guide with a summary in the form of two bullet points, each point containing the word "insurance" at least once.

<|assistant|> **The Evolution of Life Insurance Policies in the United States**

Life insurance has undergone significant changes over the years, adapting to the evolving needs of policyholders and the regulatory environment. Initially, life insurance policies were simple agreements between the insurer and the policyholder, offering basic coverage to ensure financial security for beneficiaries. Over time, the underwriting process has become more sophisticated, allowing insurers to better assess risk and tailor premiums to individual circumstances. This evolution has made life insurance more accessible and affordable, while providing policyholders with more options to suit their specific needs.

The role of underwriting in life insurance has expanded significantly, with advances in technology enabling insurers to gather and analyze more data than ever before. Underwriting now involves assessing a range of factors, from medical history to lifestyle habits, to determine the appropriate coverage and premium for each policyholder. Beneficiaries stand to gain from these advancements, as more precise underwriting leads to policies that better match the risk profile of the insured, ensuring that the coverage provided is both adequate and cost-effective. With these innovations, insurers can offer competitive premiums while maintaining their commitment to providing security for beneficiaries.

As life insurance policies have evolved, so too have the options available to policyholders and the benefits they provide. Modern life insurance products offer a variety of coverage options, from term life to whole life and universal life, each designed to meet different financial goals and needs. Policyholders can now customize their insurance plans, adding riders for additional coverage or flexible premiums to suit their changing circumstances. These innovations have significantly increased the importance of life insurance as a financial planning tool, offering both protection and investment opportunities for policyholders and their beneficiaries.

- Life insurance has evolved to offer more personalized coverage and competitive premiums.
- Modern life insurance products provide flexible options, enhancing their role in financial planning.<|end_of_text|><|user|> As a well-established Hollywood celebrity known for my dance skills in musicals, I want to explore new dance trends and innovations in choreography. Please provide me with a list of innovative dance styles or technologies in dance. Output two bullet lists, each containing at least five items. Additionally, write an introductory paragraph explaining the influence of technology on modern dance. Ensure that the entire output is in english and lowercase.

<|assistant|> technology has significantly influenced modern dance, introducing new possibilities and expanding the creative horizons for dancers and choreographers. advancements in technology have not only enriched the visual and auditory experience of performances but have also led to the emergence of new dance styles that integrate digital elements. from interactive stage designs to virtual reality experiences, technology is reshaping the way dance is created and consumed. this fusion of dance and technology allows for more immersive and engaging performances, providing audiences with innovative experiences that transcend traditional boundaries.

innovative dance styles:
- waacking: a highly expressive style characterized by fast arm movements and poses, originating from the lgbtq+ clubs of los angeles.
- krumping: an energetic street dance style that involves expressive, exaggerated movements and freestyle dance battles.
- tutting: a style inspired by the angular poses seen in ancient egyptian art, focusing on intricate hand and arm movements.
- flexing (bone breaking): a style known for its contortionist movements and illusions that create striking visuals.
- voguing: a style that combines fashion poses with angular, linear, and rigid movements, often performed in ballroom culture.

technologies in dance:
- motion capture: technology that records the movements of dancers to create digital animations or enhance live performances.
- projection mapping: a technique that projects video content onto physical surfaces, creating dynamic backgrounds and interactive environments.
- led costumes: smart clothing embedded with led lights, allowing for visually stunning performances that synchronize light with movement.
- virtual reality (vr): immersive experiences that transport audiences to a digital world, offering new perspectives and interactions in dance.
- augmented reality (ar): overlays digital content onto the real world, enabling interactive performances where virtual elements blend with live dancers.<|end_of_text|><|user|> Describe in 8 sentences the most memorable race you had against Jack Johnson. Each sentence must contain the letter 'r' at least 5 times. Use the word 'speed' exactly 3 times in your entire response. End your response with the phrase 'Victory is sweet!'

<|assistant|> Racing against Jack Johnson, I remember the roaring crowd urging us forward with fervor. The track seemed riddled with hurdles, each turn requiring precision and incredible speed. Our sneakers barely skimmed the ground, reverberating with resounding rhythm. Jack's rapid strides were remarkable, matching my speed stride for stride. My heart raced, thrilled by the relentless pursuit of victory. Every breath felt sharp, air rushing in and out as if trying to keep pace with our speed. The finish line approached, our rivalry reaching a fever pitch, hearts pounding with every rapid heartbeat. Crossing the line first, I felt the sweet sting of triumph, and thought, "Victory is sweet!"<|end_of_text|><|user|> Write the lyrics for a song consisting of exactly 8 sentences that celebrate the fusion of ancient folklore with contemporary life. Include the title of the song wrapped in double angular brackets, i.e. <<title>> and repeat the title twice within the lyrics.

<|assistant|> <<Echoes of the Past>>

In the city lights, where legends unfold,  
Ancient whispers blend with stories told.  
<<Echoes of the Past>> sing through the night,  
Guiding our steps with a gentle light.  

From the tales of old, new dreams arise,  
Weaving the magic beneath modern skies.  
<<Echoes of the Past>> in every heart,  
Binding us together, never apart.<|end_of_text|><|user|> Compose an educational article about the significance of Islamic art in cultural heritage. The article should include the following constraints:
- Use no more than 5 words in all capital letters to emphasize key concepts.
- Incorporate exactly 3 bullet points in a list to highlight the main elements of Islamic art's impact on culture.
- The article must be written in English and should maintain a respectful and informative tone suitable for a broad audience interested in cultural studies.

<|assistant|> Islamic art holds a profound significance in cultural heritage, reflecting the rich tapestry of history, religion, and aesthetics that spans across various regions and centuries. Its influence is not confined to religious contexts; instead, it permeates diverse aspects of life, highlighting the unity and diversity within the Islamic world. This art form is characterized by its unique features and contributions, which have left a lasting impact on global culture.

One of the most striking aspects of Islamic art is its emphasis on GEOMETRIC PATTERNS and intricate designs. These patterns are not merely decorative; they embody a deep philosophical and spiritual symbolism, representing the infinite nature of creation. The use of symmetry and repetition in these patterns conveys a sense of balance and harmony, reflecting the artist's contemplation of the universe.

Islamic art also places significant importance on CALLIGRAPHY, regarded as the highest form of artistic expression in many Islamic cultures. The beautiful scripts used to transcribe the Quran and other religious texts have transcended their original purpose, inspiring secular art and design. The elegance and precision of Islamic calligraphy continue to captivate audiences worldwide, serving as a bridge between the sacred and the artistic.

The impact of Islamic art on cultural heritage is evident in various ways:

- **Architecture:** Majestic structures like the Alhambra and the Taj Mahal showcase the artistic ingenuity and cultural exchange within the Islamic world.
- **Textiles:** Richly woven fabrics with elaborate patterns have influenced fashion and design far beyond their origins.
- **Ceramics:** Distinctive styles and techniques in pottery illustrate the fusion of local traditions with Islamic aesthetics.

In conclusion, Islamic art is a vital component of cultural HERITAGE, offering insights into the shared human experience while celebrating the diversity of expression. Its enduring legacy is a testament to the power of art to transcend boundaries and connect cultures. Through its beauty and complexity, Islamic art continues to inspire and educate, ensuring its place in the global cultural landscape for generations to come.<|end_of_text|><|user|> Write a fan chant for the Punjab Panthers to be used during their matches. Please use an enthusiastic tone and ensure that the chant includes the keywords "roar," "victory," and "spirit." Make sure that the number of words in all capital letters is exactly 4, and use no commas in the chant.

<|assistant|> Roar Panthers roar with MIGHT and PRIDE  
Feel the spirit on our side  
Victory is OURS tonight  
Punjab Panthers FIGHT FIGHT FIGHT!<|end_of_text|><|user|> Write a detailed 4-paragraph artist's statement that explores the inspiration and process behind my paintings of supermodels, emphasizing their beauty and elegance. Begin the first paragraph with the word "Inspired" and conclude the final paragraph with the word "timelessness".

<|assistant|> Inspired by the allure and poise of supermodels, my artistic journey delves into the vibrant world of high fashion and its embodiment of contemporary beauty. The supermodel is not merely a subject but a muse that encapsulates the quintessence of elegance and confidence. Each painting is a celebration of their unique personas, capturing their essence in a way that transcends the ephemeral nature of fashion trends. By focusing on the supermodel, I aim to channel the charisma and strength they exude, inviting viewers to explore the deeper narratives that lie beneath the surface of their captivating visages.

In my creative process, I begin by immersing myself in the stories of the supermodels I paint. I study their runway walks, editorial shoots, and personal interviews to gain a holistic understanding of their individuality. This preparation is crucial, as it informs the choices I make regarding composition, color palette, and brushwork. I strive to capture the dynamism of their movements, the subtlety of their expressions, and the luminosity of their presence. Through this meticulous approach, I aim to create paintings that not only reflect their physical beauty but also convey the intangible qualities that make them icons of elegance.

The interplay of light and shadow plays a pivotal role in my artwork, as it allows me to highlight the contours and depths of the supermodels' features. I often employ a combination of oil and acrylic paints to achieve a textural richness that adds dimension to the canvas. The application of layers and glazes helps in creating a sense of depth, while delicate brushstrokes capture the fleeting moments of grace and poise. By intertwining realism with abstraction, my paintings evoke an emotional response, inviting the audience to appreciate the interplay between the seen and unseen elements of beauty.

Ultimately, my paintings of supermodels are an homage to the timelessness of beauty and elegance. Through my art, I aspire to transcend the conventional boundaries of portraiture, offering a fresh perspective on the enigmatic allure of these figures. Each piece is a testament to the enduring impact of their presence in the world of fashion and beyond. By capturing their essence on canvas, I endeavor to create a lasting impression that resonates with viewers, reminding them of the power and grace that define true timelessness.<|end_of_text|><|user|> Please integrate the following question into your response and repeat the prompt: How has the prison education system influenced the lives of former inmates, and what are some of the broader societal benefits of investing in these programs?

<|assistant|> The question at hand is: How has the prison education system influenced the lives of former inmates, and what are some of the broader societal benefits of investing in these programs? The prison education system has had a profound impact on the lives of former inmates by providing them with the skills and knowledge necessary to successfully reintegrate into society. Education programs in prisons often lead to lower recidivism rates, as individuals are better equipped to find employment and engage in productive activities post-release. Furthermore, former inmates who have participated in education programs are more likely to contribute positively to their communities.

In terms of broader societal benefits, investing in prison education programs can lead to significant cost savings for the criminal justice system by reducing the rates of re-incarceration. Education also promotes social equity by offering second chances and reducing the cycle of disadvantage. Additionally, by transforming the lives of former inmates, these programs contribute to safer communities and a more skilled workforce. Thus, the positive ripple effects of investing in prison education are far-reaching.<|end_of_text|><|user|> Rewrite the following analysis into a JSON format, ensuring each sentence is in a separate field named 'sentence_1', 'sentence_2', etc. The analysis must contain exactly five sentences, written in English, and all sentences should start with a capital letter:

The Soviet Union exercised significant influence over Eastern Europe during the Cold War. This influence was manifested through both political and military means. Many Eastern European countries were compelled to adopt communism as their form of government. The Soviet Union also impacted the region's economies by integrating them into the Eastern Bloc's economic system. However, this dominance faced resistance, leading to uprisings and eventual reforms.

<|assistant|> ```json
{
  "sentence_1": "The Soviet Union exercised significant influence over Eastern Europe during the Cold War.",
  "sentence_2": "This influence was manifested through both political and military means.",
  "sentence_3": "Many Eastern European countries were compelled to adopt communism as their form of government.",
  "sentence_4": "The Soviet Union also impacted the region's economies by integrating them into the Eastern Bloc's economic system.",
  "sentence_5": "However, this dominance faced resistance, leading to uprisings and eventual reforms."
}
```<|end_of_text|><|user|> Provide two separate summaries of the latest advancements in infectious disease containment strategies. Each summary should be exactly 3 paragraphs long and include one case study of an innovative strategy. Each paragraph should have a title in bold.

<|assistant|> **Summary 1: Enhanced Surveillance and Response Systems**

**Improved Data Integration**  
Recent advancements in infectious disease containment have seen a significant improvement in data integration methods. With the advent of advanced digital technologies, health agencies worldwide are now able to compile, process, and analyze vast amounts of data from diverse sources. These include hospital records, laboratory results, and even social media trends, which allow for more accurate predictions and timely responses to emerging outbreaks. By harnessing big data analytics, health authorities can effectively monitor disease spread and implement targeted interventions.

**AI-Powered Predictive Models**  
Artificial intelligence (AI) has revolutionized the way we predict and manage infectious disease outbreaks. AI-powered models can simulate various outbreak scenarios, offering insights into potential disease spread patterns. These models assist in identifying high-risk areas and populations, enabling tailored containment strategies such as vaccination campaigns and resource allocation. The real-time adaptability of AI models has proven crucial in minimizing the impact of diseases before they reach pandemic levels.

**Case Study: Taiwan's Digital Contact Tracing**  
Taiwan has implemented an innovative digital contact tracing system as part of its infectious disease containment efforts. This system leverages mobile technology and data analytics to track and trace potential contacts of infected individuals. Through a centralized health monitoring platform, authorities can swiftly notify and isolate contacts, significantly reducing transmission rates. Taiwan's success in containing COVID-19 has been largely attributed to this efficient and privacy-conscious approach, setting a precedent for other nations seeking to enhance their contact tracing capabilities.

**Summary 2: Community Engagement and Decentralized Approaches**

**Community-Centric Health Initiatives**  
Increasingly, public health strategies are focusing on community engagement to bolster infectious disease containment. By involving local communities in health education and prevention efforts, authorities can foster a sense of ownership and responsibility. Community health workers play a pivotal role in disseminating accurate information, debunking myths, and encouraging preventive behaviors such as vaccination and hygiene practices. This grassroots approach ensures that containment strategies are culturally sensitive and widely accepted.

**Decentralized Healthcare Systems**  
Decentralization of healthcare services has emerged as a key strategy in managing infectious diseases. By distributing healthcare resources and decision-making closer to affected communities, response times are reduced, and interventions can be more effectively tailored to local needs. This approach empowers local health facilities and leaders, making them instrumental in early detection and rapid response to outbreaks. Decentralized systems also alleviate the burden on central hospitals, ensuring that healthcare remains accessible to all.

**Case Study: Senegal's Rapid Diagnostic Testing**  
Senegal's deployment of rapid diagnostic testing (RDT) for diseases such as Ebola and COVID-19 exemplifies a successful decentralized approach. By training local health workers to administer RDTs and equipping regional healthcare centers, Senegal has dramatically improved early detection rates and response times. This strategy not only curbs the spread of disease but also builds local capacity and resilience. Senegal's model highlights the importance of equipping communities with the tools and knowledge necessary for effective disease management.
```

</details>

---

### Sample 6 — `math_competition`

**ID:** gp446 | **Sources:** 4 raw samples | **Padding:** 288 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> The altitude of a right prism is 1 m, its base is a rhombus with side equal to 2 m and an acute angle of $30^\circ$. A secant plane is drawn through a side of the base which is at an angle of $60^\circ$ to the plane of the base. Find the area of the section.

<|assistant|> Let  

* the altitude (height) of the right prism be \(h=1\;\mathrm m\);
* the base be a rhombus of side \(s=2\;\mathrm m\) whose acute angle is \(\alpha =30^{\circ}\).

---

### 1.  Geometry of the base  

Place the base in the plane \(z=0\).  
Choose the side through which the secant plane passes as the segment  

\[
AB:\; A(0,0,0),\qquad B(2,0,0)
\]

(the side \(AB\) lies on the \(x\)-axis).  
Because the altitude of the rhombus equals \(s\sin\alpha =2\sin30^{\circ}=1\), the opposite vertices are  

\[
D(\sqrt3,\,1,0), \qquad C(2+\sqrt3,\,1,0).
\]

The other two lateral faces are the vertical rectangles over the edges
\(AD\) and \(BC\).

---

### 2.  Equation of the secant plane  

The required plane contains the line \(AB\) and makes an angle
\(\theta =60^{\circ}\) with the base plane \(z=0\).
If \(\mathbf n\) is a normal vector of the plane, then  

\[
\mathbf n\cdot(0,0,1)=|\mathbf n|\cos\theta \quad\Longrightarrow\quad 
n_z=|\mathbf n|\cos60^{\circ}= \frac{|\mathbf n|}{2}.
\]

Since the plane contains \(AB\) (direction \((1,0,0)\)), the normal must be orthogonal to this direction, i.e. \(n_x=0\).
Hence \(\mathbf n=(0,n_y,n_z)\) with \( n_y^2 =3n_z^2\); we may choose  

\[
\mathbf n=(0,-\sqrt3,\,1).
\]

Using point \(A\) gives the plane equation  

\[
-\sqrt3\,y+z=0\qquad\Longrightarrow\qquad z=\sqrt3\,y .
\]

---

### 3.  Intersection of the plane with the prism  

A point of the prism satisfies \((x,y)\) inside the rhombus and \(0\le z\le1\).
With \(z=\sqrt3 y\) we must have  

\[
0\le \sqrt3 y\le1\;\Longrightarrow\;0\le y\le\frac1{\sqrt3}.
\]

For a fixed \(y\) the rhombus gives  

\[
x_{\min}= \sqrt3\,y,\qquad 
x_{\max}=2+\sqrt3\,y,
\]

so the width in the \(x\)-direction is constantly \(x_{\max}-x_{\min}=2\).

Therefore the orthogonal projection of the section onto the base is the
rectangle

\[
\{(x,y):0\le y\le 1/\sqrt3,\; \sqrt3\,y\le x\le2+\sqrt3\,y\}
\]

with area  

\[
A_{\text{proj}} = 2\cdot\frac{1}{\sqrt3}= \frac{2}{\sqrt3}.
\]

---

### 4.  Area of the slanted section  

If a plane makes an angle \(\theta\) with the horizontal, its area is the
area of its orthogonal projection divided by \(\cos\theta\).  
Here \(\cos60^{\circ}= \frac12\), thus  

\[
A_{\text{section}}=\frac{A_{\text{proj}}}{\cos60^{\circ}}
               =\frac{ \frac{2}{\sqrt3} }{ \frac12 }
               =\frac{4}{\sqrt3}
               =\frac{4\sqrt3}{3}\;\text{m}^{2}.
\]

(One can also compute the area directly as \(|\overrightarrow{AB}\times\overrightarrow{AE}|\)
with \(\overrightarrow{AB}=(2,0,0)\) and \(\overrightarrow{AE}=(1,\tfrac1{\sqrt3},1)\);
the cross‑product magnitude is \(\frac{4}{\sqrt3}\) as above.)

---

\[
\boxed{\displaystyle \frac{4\sqrt3}{3}\ \text{m}^2}
\]<|end_of_text|><|user|> Joey is 1 year older than Chloe, and Zoe is exactly 1 year old today. Today is the first of the 9 birthdays on which Chloe's age will be an integral multiple of Zoe's age. What will be the sum of the two digits of Joey's age the next time his age is a multiple of Zoe's age?

<|assistant|> The ages are counted on birthdays, so all three have a birthday on the same day.

Let  

* Zoe’s age today = 1.  
* Chloe’s age today = \(C\).  
* Joey’s age today = \(C+1\) (since Joey is 1 year older).

---

### When is Chloe’s age a multiple of Zoe’s?

On the birthday that occurs \(t\) years from now  

\[
\text{Zoe’s age}=1+t,\qquad \text{Chloe’s age}=C+t .
\]

We need  

\[
C+t\equiv0\pmod{1+t}.
\]

Write \(d=1+t\;(>1)\). Then \(t=d-1\) and the condition becomes  

\[
C+d-1\equiv0\pmod d\iff C-1\equiv0\pmod d .
\]

Thus \(d\) must be a divisor of \(C-1\).  
Conversely, each divisor \(d\) of \(C-1\) gives a birthday (with \(t=d-1\)) where the ratio is integral.

Hence the number of such birthdays equals the number of positive divisors of \(C-1\).

We are told there are **9** such birthdays (including today).  
Therefore

\[
\tau(C-1)=9,
\]

where \(\tau(n)\) is the divisor‑count function.

The possibilities for an integer with exactly 9 divisors are  

* \(n=p^8\) (a prime to the 8th power), or  
* \(n=p^2q^2\) (product of two distinct primes each squared).

The first case gives \(C-1\ge2^8=256\Rightarrow C\ge257\), which would make Joey’s later age far beyond two digits.  
So we use the second case:

\[
C-1=p^2q^2=(pq)^2 .
\]

Let \(p<q\) be distinct primes. Then  

\[
C=(pq)^2+1 .
\]

To keep Joey’s next multiple a two‑digit number, \(C\) itself must be at most about 90. The smallest product of two distinct primes is \(2\cdot3=6\):

\[
C=(6)^2+1=36+1=37 .
\]

Indeed, \(C-1=36=2^2\cdot3^2\) has exactly nine divisors, giving the required nine birthdays.

---

### Joey’s next multiple of Zoe’s age

Joey’s age \(J\) at a birthday \(t\) years from now is  

\[
J(t)=C+1+t.
\]

We need \(J(t)\) to be a multiple of Zoe’s age \(1+t\).  As before, letting \(d=1+t\),

\[
J(t)\equiv0\pmod d\iff C\equiv0\pmod d .
\]

Thus \(d\) must be a divisor of \(C\).  
Since \(C=37\) is prime, its only divisor \(>1\) is \(d=37\).

Hence the **next** time (the smallest \(t>0\)) is when  

\[
d=37\;\Longrightarrow\;t=d-1=36 .
\]

At that time  

\[
\text{Zoe’s age}=1+36=37,\\
\text{Joey’s age}=C+1+t=38+36=74 .
\]

Joey’s age is \(74\), a two‑digit number.  
The sum of its digits is  

\[
7+4=11 .
\]

\[
\boxed{11}
\]<|end_of_text|><|user|> Ellie has two lists, each consisting of 6 consecutive positive integers. The smallest integer in the first list is \(a\), the smallest integer in the second list is \(b\), and \(a < b\). She makes a third list which consists of 36 integers formed by multiplying each number from the first list with each number in the second list. If:
1. The integer 49 appears in the third list,
2. There is no number in the third list that is a multiple of 64, and
3. There is at least one number in the third list that is larger than 75,
Determine all possible pairs \((a, b)\).

<|assistant|> The factorisation \(49=7^2\) forces the product \(49\) to arise either as  

* \(1\cdot 49\) (so the first list must contain \(1\) and the second list must contain \(49\)), or  
* \(7\cdot 7\) (so both lists must contain the integer \(7\)).  

---

### 1.  Case \(1\cdot 49\)

* To have \(1\) in the first list we need \(a=1\).  
* To have \(49\) in the second list we need \(b\le 49\le b+5\), i.e. \(b\in\{44,45,46,47,48,49\}\).

If \(b=44,45,46,47\) or \(48\) the second list contains \(48\); together with the element \(4\) from the first list we obtain the product \(4\cdot48=192\), which is a multiple of \(64\).  
Hence those values are forbidden by condition 2.

For \(b=49\) the second list is \(\{49,50,51,52,53,54\}\); none of the products with \(\{1,2,3,4,5,6\}\) is a multiple of \(64\), and there are products larger than \(75\) (e.g. \(6\cdot49=294\)).  

Thus one admissible pair is  

\[
( a , b ) = (1,49).
\]

---

### 2.  Case \(7\cdot 7\)

Both lists must contain \(7\).  
Hence

\[
a\le7\le a+5,\qquad b\le7\le b+5,
\]

so \(a,b\in\{2,3,4,5,6,7\}\) with \(a<b\).  
The possible ordered pairs are  

\[
(2,3),(2,4),(2,5),(2,6),(2,7),
(3,4),(3,5),(3,6),(3,7),
(4,5),(4,6),(4,7),
(5,6),(5,7),
(6,7).
\]

#### Condition 3 (product \(>75\))
A product exceeds \(75\) iff \((a+5)(b+5)>75\).  
The pairs that satisfy this are  

\[
(2,6),(2,7),(3,5),(3,6),(3,7),
(4,5),(4,6),(4,7),(5,6),(5,7),(6,7).
\]

The remaining four pairs \((2,3),(2,4),(2,5),(3,4)\) are eliminated.

#### Condition 2 (no multiple of \(64\))
The only possible multiple of \(64\) among products of numbers \(\le12\) is \(64=8\cdot8\).  
Hence we must avoid having the integer \(8\) in **both** lists.

Checking the above eleven pairs, both lists contain \(8\) for every pair except \((2,6)\) and \((2,7)\).  
Thus only these two survive condition 2.

Both of them indeed give the required factor \(7\cdot7=49\) (since \(7\) belongs to each list) and contain a product larger than \(75\) (\(7\cdot11=77\) and \(7\cdot12=84\) respectively).

Hence the admissible pairs from this case are  

\[
(2,6)\;\text{and}\;(2,7).
\]

---

### 3.  All admissible ordered pairs

Collecting the results from the two cases, the only ordered pairs \((a,b)\) that satisfy all three conditions are

\[
\boxed{(1,49),\;(2,6),\;(2,7)}.
\]<|end_of_text|><|user|> Compute the sum \( \sin^2 0^\circ + \sin^2 1^\circ + \sin^2 2^\circ + \sin^2 3^\circ + \sin^2 4^\circ + \ldots + \sin^2 99^\circ \).

<|assistant|> \[
S=\sum_{k=0}^{99}\sin ^2k^{\circ}
\]

---

### 1.  Use the double–angle identity  

\[
\sin ^2\theta=\frac{1-\cos 2\theta}{2}
\]

Hence  

\[
S=\frac12\sum_{k=0}^{99}\bigl(1-\cos 2k^{\circ}\bigr)
   =\frac12\Bigl(100-\sum_{k=0}^{99}\cos 2k^{\circ}\Bigr).
\]

Set  

\[
C=\sum_{k=0}^{99}\cos 2k^{\circ}.
\]

---

### 2.  Compute \(C\) with a geometric‑series trick  

\[
\cos 2k^{\circ}= \Re\!\bigl(e^{i2k^{\circ}}\bigr).
\]

Thus  

\[
C=\Re\!\Bigl(\sum_{k=0}^{99}e^{i2k^{\circ}}\Bigr)
   =\Re\!\Bigl(\frac{1-e^{i200^{\circ}}}{1-e^{i2^{\circ}}}\Bigr).
\]

Multiplying numerator and denominator by the conjugate of the denominator gives  

\[
C=\frac{\sin(100^{\circ})\cos(99^{\circ})}{\sin 1^{\circ}} .
\]

Use \(\sin(100^{\circ})=\sin(80^{\circ})\) and the product–to–sum formula:

\[
\sin 100^{\circ}\cos 99^{\circ}
   =\frac12\bigl[\sin(199^{\circ})+\sin 1^{\circ}\bigr]
   =\frac12\bigl[-\sin 19^{\circ}+\sin 1^{\circ}\bigr].
\]

Therefore  

\[
C=\frac{\frac12\bigl[-\sin 19^{\circ}+\sin 1^{\circ}\bigr]}{\sin 1^{\circ}}
  =\frac12\Bigl(1-\frac{\sin 19^{\circ}}{\sin 1^{\circ}}\Bigr).
\]

---

### 3.  Finish the sum  

\[
\begin{aligned}
S&=\frac12\Bigl(100-C\Bigr)\\[2mm]
 &=\frac12\Bigl(100-\tfrac12\bigl(1-\frac{\sin 19^{\circ}}{\sin 1^{\circ}}\bigr)\Bigr)\\[2mm]
 &=\frac{199}{4}+\frac{\sin 19^{\circ}}{4\sin 1^{\circ}} .
\end{aligned}
\]

---

### 4.  Numerical value (optional)

\[
\sin 19^{\circ}\approx0.325568154,\qquad 
\sin 1^{\circ}\approx0.017452406,
\]

so  

\[
S\approx\frac{199}{4}+ \frac{0.325568154}{4\cdot0.017452406}
   \approx 49.75+4.6636\approx 54.4136.
\]

---

\[
\boxed{\displaystyle S=\frac{199}{4}+\frac{\sin 19^{\circ}}{4\sin 1^{\circ}}}
\]

(≈ 54.4136).
```

</details>

---

### Sample 7 — `math_hard`

**ID:** gp490 | **Sources:** 4 raw samples | **Padding:** 358 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> Consider the parametric equations defined by:
\[
x = \frac{3t}{1 + t^2}, \quad y = \frac{3(1 - t^2)}{1 + t^2}, \quad t \in \mathbb{R}.
\]
(a) Show that these parametric equations describe a curve in the Cartesian plane. Identify the curve and provide its Cartesian equation.
(b) Determine the range of \( t \) values that correspond to the upper and lower semicircles of the curve.
(c) Compute the derivative \( \frac{dy}{dx} \) as a function of \( t \) and evaluate it at \( t = 1 \).
(d) Verify that the curve is traced exactly once as \( t \) varies over all real numbers.
(a) To eliminate the parameter \( t \) and find the Cartesian equation, observe the following:
\[
x^2 + y^2 = \left( \frac{3t}{1 + t^2} \right)^2 + \left( \frac{3(1 - t^2)}{1 + t^2} \right)^2 = \frac{9t^2 + 9(1 - 2t^2 + t^4)}{(1 + t^2)^2} = \frac{9(1 + t^2)}{(1 + t^2)^2} = \frac{9}{1 + t^2}.
\]
However, notice that:
\[
y = \frac{3(1 - t^2)}{1 + t^2} = \frac{3(1 + t^2) - 6t^2}{1 + t^2} = 3 - \frac{6t^2}{1 + t^2}.
\]
From \( x^2 + y^2 = \frac{9}{1 + t^2} \), solve for \( \frac{1}{1 + t^2} \):
\[
\frac{1}{1 + t^2} = \frac{x^2 + y^2}{9}.
\]
Substitute into the expression for \( y \):
\[
y = 3 - 6t^2 \cdot \frac{x^2 + y^2}{9} = 3 - \frac{2}{3} t^2 (x^2 + y^2).
\]
This suggests a more straightforward approach. Instead, note that:
\[
x^2 + \left( \frac{y}{3} + 1 \right)^2 = x^2 + \left( \frac{3(1 - t^2)}{3(1 + t^2)} + 1 \right)^2 = x^2 + \left( \frac{1 - t^2 + 1 + t^2}{1 + t^2} \right)^2 = x^2 + \left( \frac{2}{1 + t^2} \right)^2.
\]
This does not simplify directly. Instead, observe:
\[
\frac{x}{3} = \frac{t}{1 + t^2}, \quad \frac{y + 3}{6} = \frac{1}{1 + t^2}.
\]
Let \( u = \frac{t}{1 + t^2} \) and \( v = \frac{1}{1 + t^2} \). Then \( \frac{x}{3} = u \) and \( \frac{y + 3}{6} = v \), with \( u^2 + v^2 = \frac{t^2 + 1}{(1 + t^2)^2} = \frac{1}{1 + t^2} = v \). Thus:
\[
\left( \frac{x}{3} \right)^2 + \left( \frac{y + 3}{6} \right)^2 = \frac{y + 3}{6}.
\]
This is the equation of a circle. To simplify:
\[
\frac{x^2}{9} + \frac{(y + 3)^2}{36} = \frac{y + 3}{6}.
\]
Multiply both sides by 36:
\[
4x^2 + (y + 3)^2 = 6(y + 3).
\]
Expand and rearrange:
\[
4x^2 + y^2 + 6y + 9 = 6y + 18,
\]
\[
4x^2 + y^2 = 9,
\]
\[
\frac{x^2}{\left( \frac{3}{2} \right)^2} + \frac{y^2}{3^2} = 1.
\]
Thus, the curve is an ellipse centered at the origin with semi-major axis 3 (along the \( y \)-axis) and semi-minor axis \( \frac{3}{2} \) (along the \( x \)-axis).

(b) To determine the range of \( t \) for the upper and lower semicircles, solve for \( t \) in terms of \( y \):
\[
y = \frac{3(1 - t^2)}{1 + t^2},
\]
\[
y(1 + t^2) = 3(1 - t^2),
\]
\[
y + y t^2 = 3 - 3 t^2,
\]
\[
(y + 3) t^2 = 3 - y,
\]
\[
t^2 = \frac{3 - y}{y + 3}.
\]
The upper semicircle corresponds to \( y \geq 0 \), and the lower semicircle corresponds to \( y \leq 0 \). For \( y \geq 0 \), \( t^2 \geq 0 \) implies \( \frac{3 - y}{y + 3} \geq 0 \), so \( y \in [0, 3) \). Thus, \( t \) is unrestricted (all real \( t \) give \( y \geq 0 \)). For \( y \leq 0 \), \( t^2 \geq 0 \) implies \( \frac{3 - y}{y + 3} \geq 0 \), but \( y \leq 0 \) gives \( y + 3 \geq 3 > 0 \), so \( 3 - y \geq 0 \) always holds. Thus, \( y \in (-3, 0] \), and \( t \) is again unrestricted. However, note that as \( t \to \pm \infty \), \( x \to 0 \) and \( y \to -3 \). The upper semicircle (\( y \geq 0 \)) is traced for all real \( t \), while the lower semicircle (\( y \leq 0 \)) is also traced for all real \( t \). This suggests the entire ellipse is traced as \( t \) varies over \( \mathbb{R} \).

(c) Compute \( \frac{dy}{dx} \) using the chain rule:
\[
\frac{dy}{dx} = \frac{dy/dt}{dx/dt}.
\]

<|assistant|> First, compute \( \frac{dx}{dt} \) and \( \frac{dy}{dt} \):
\[
x = \frac{3t}{1 + t^2},
\]
\[
\frac{dx}{dt} = \frac{3(1 + t^2) - 3t(2t)}{(1 + t^2)^2} = \frac{3 - 3t^2}{(1 + t^2)^2} = \frac{3(1 - t^2)}{(1 + t^2)^2}.
\]
\[
y = \frac{3(1 - t^2)}{1 + t^2},
\]
\[
\frac{dy}{dt} = \frac{-6t(1 + t^2) - 3(1 - t^2)(2t)}{(1 + t^2)^2} = \frac{-6t - 6t^3 - 6t + 6t^3}{(1 + t^2)^2} = \frac{-12t}{(1 + t^2)^2}.
\]
Thus:
\[
\frac{dy}{dx} = \frac{-12t / (1 + t^2)^2}{3(1 - t^2) / (1 + t^2)^2} = \frac{-12t}{3(1 - t^2)} = \frac{-4t}{1 - t^2}.
\]
At \( t = 1 \):
\[
\frac{dy}{dx} = \frac{-4(1)}{1 - 1^2} = \frac{-4}{0},
\]
which is undefined, indicating a vertical tangent.

(d) To verify that the curve is traced exactly once, show that the mapping \( t \mapsto (x(t), y(t)) \) is injective. Suppose \( t_1 \neq t_2 \) but \( (x(t_1), y(t_1)) = (x(t_2), y(t_2)) \). Then:
\[
\frac{3t_1}{1 + t_1^2} = \frac{3t_2}{1 + t_2^2},
\]
\[
t_1(1 + t_2^2) = t_2(1 + t_1^2),
\]
\[
t_1 + t_1 t_2^2 = t_2 + t_2 t_1^2,
\]
\[
t_1 - t_2 = t_2 t_1^2 - t_1 t_2^2 = t_1 t_2 (t_1 - t_2).
\]
If \( t_1 \neq t_2 \), then divide by \( t_1 - t_2 \):
\[
1 = t_1 t_2.
\]
Now, from \( y(t_1) = y(t_2) \):
\[
\frac{3(1 - t_1^2)}{1 + t_1^2} = \frac{3(1 - t_2^2)}{1 + t_2^2},
\]
\[
(1 - t_1^2)(1 + t_2^2) = (1 - t_2^2)(1 + t_1^2),
\]
\[
1 + t_2^2 - t_1^2 - t_1^2 t_2^2 = 1 + t_1^2 - t_2^2 - t_1^2 t_2^2,
\]
\[
t_2^2 - t_1^2 = t_1^2 - t_2^2,
\]
\[
2(t_2^2 - t_1^2) = 0,
\]
\[
t_2^2 = t_1^2.
\]
Thus, \( t_2 = \pm t_1 \). If \( t_2 = t_1 \), contradiction. If \( t_2 = -t_1 \), then from \( 1 = t_1 t_2 \), \( 1 = -t_1^2 \), which has no real solutions. Therefore, \( t_1 = t_2 \), and the curve is traced exactly once.<|end_of_text|><|user|> A local art class is creating a mural inspired by Wayne Thiebaud's famous painted cakes. The mural includes a large circular cake design with a radius of 10 feet. The students plan to paint a shaded sector of the cake to represent a slice of the cake. The shaded sector has a central angle of 72 degrees. 

1. What is the area of the shaded sector in square feet?
2. If the students decide to paint the curved edge of the sector (the arc), how long is the arc in feet?

<|assistant|> To solve this problem, we will use concepts from Geometry and Trigonometry.

1. **Area of the shaded sector:**

   The area $ A $ of a sector of a circle is given by the formula:

   $$
   A = \frac{\theta}{360} \cdot \pi r^2
   $$

   where:
   - $ \theta $ is the central angle in degrees,
   - $ r $ is the radius of the circle.

   Given:
   - $ \theta = 72^\circ $,
   - $ r = 10 $ feet.

   Substituting into the formula:

   $$
   A = \frac{72}{360} \cdot \pi (10)^2
   $$

   Simplify the fraction:

   $$
   A = \frac{1}{5} \cdot \pi \cdot 100
   $$

   $$
   A = \frac{1}{5} \cdot 100\pi
   $$

   $$
   A = 20\pi
   $$

   So, the area of the shaded sector is $ 20\pi $ square feet.

2. **Length of the arc:**

   The length $ L $ of an arc of a circle is given by the formula:

   $$
   L = \frac{\theta}{360} \cdot 2\pi r
   $$

   Using the same values for $ \theta $ and $ r $:

   $$
   L = \frac{72}{360} \cdot 2\pi (10)
   $$

   Simplify the fraction:

   $$
   L = \frac{1}{5} \cdot 20\pi
   $$

   $$
   L = 4\pi
   $$

   So, the length of the arc is $ 4\pi $ feet.

In conclusion:
- The area of the shaded sector is $ 20\pi $ square feet.
- The length of the arc is $ 4\pi $ feet.<|end_of_text|><|user|> A local park has a rectangular garden with a length that is 3 meters longer than its width. If the perimeter of the garden is 34 meters, what are the dimensions of the garden?

<|assistant|> To solve this problem, we need to use the information given to set up an equation and then solve for the dimensions of the garden.

1. **Identify Variables**:
   Let \( w \) be the width of the garden in meters.
   Since the length is 3 meters longer than the width, the length \( l \) can be expressed as \( l = w + 3 \).

2. **Establish Relationships**:
   The perimeter \( P \) of a rectangle is given by the formula:
   \[
   P = 2l + 2w
   \]
   We know the perimeter is 34 meters, so we can write:
   \[
   2l + 2w = 34
   \]

3. **Substitute the Length**:
   Substitute \( l = w + 3 \) into the perimeter equation:
   \[
   2(w + 3) + 2w = 34
   \]

4. **Simplify the Equation**:
   Distribute the 2 in the equation:
   \[
   2w + 6 + 2w = 34
   \]
   Combine like terms:
   \[
   4w + 6 = 34
   \]

5. **Solve for \( w \)**:
   Subtract 6 from both sides:
   \[
   4w = 28
   \]
   Divide both sides by 4:
   \[
   w = 7
   \]

6. **Find the Length**:
   Substitute \( w = 7 \) back into the expression for the length:
   \[
   l = w + 3 = 7 + 3 = 10
   \]

Therefore, the dimensions of the garden are:
- Width: 7 meters
- Length: 10 meters<|end_of_text|><|user|> A class of 24 students each flipped a coin 10 times as part of an experiment to study probability. They recorded how many times the coin landed on heads. Altogether, the class got 112 heads. What is the experimental probability of getting heads in one coin flip based on this experiment?

<|assistant|> Step 1: Find the total number of coin flips.  
Each student flipped the coin 10 times, and there are 24 students.  
So, we multiply 24 by 10.  
24 × 10 = 240 total flips.  

Step 2: Find the number of times heads occurred.  
The class recorded 112 heads in total.  

Step 3: Calculate the experimental probability.  
Experimental probability is the number of successful outcomes (heads) divided by the total number of trials (flips).  
So, we divide 112 by 240.  
112 ÷ 240 = 0.4666...  

Step 4: Round the result to two decimal places.  
0.4666... is approximately 0.47.  

So, the experimental probability of getting heads in one coin flip is 0.47.
```

</details>

---

### Sample 8 — `human_preferred`

**ID:** gp164 | **Sources:** 1 raw samples | **Padding:** 282 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> 有没有一些故事充斥着相互的矛盾和冲突，每个人都做出了自己认为正确的选择，但是将故事线或者说从更宏观的层面来看，看似每个人正确的结局却导致了整体上最大的悲剧。以及每个人在生命最后的抉择，都与其身份和描写所矛盾，而有一种强烈的冲击感。这是否表明人们更喜欢某些残缺的故事，类似于华纳的蝙蝠侠和小丑的故事，深刻的让人们去反思传统的道德在现代社会中是否还适用？什么才是一个好人？什么才是一个坏人

<|assistant|> 你提出的这个主题非常深刻，确实可以在许多文学和电影作品中找到相互矛盾的角色和冲突的故事情节。这类故事不仅挑战了传统的道德观念，也引发观众对“好人”和“坏人”界限的深思。以下是一些符合你描述的故事和主题，探讨个人选择与整体悲剧之间的矛盾。

### 1. **《蝙蝠侠：黑暗骑士》**
  - **故事背景**：这部电影展示了蝙蝠侠与小丑之间的对立。蝙蝠侠试图维护城市的秩序，而小丑则通过混乱来挑战这一秩序。
  - **矛盾冲突**：每个角色都在为自己的信念而战。蝙蝠侠认为要通过恐惧来维持正义，而小丑则认为人性本恶，试图揭露这一点。最终，蝙蝠侠为了保护哈维·丹特的形象而选择牺牲自己的名声，这一选择在表面上是正确的，但却导致了更大的悲剧。

### 2. **《1984》**
  - **故事背景**：乔治·奥威尔的反乌托邦小说，描绘了一个极权统治下的人类生存状态。
  - **矛盾冲突**：温斯顿·史密斯试图反抗体制，追求真相和自由。他的选择虽然是出于对人性的追求，但最终却导致了个人的彻底毁灭和对体制的无奈屈服。这种选择的悲剧性在于他明知选择的后果，却依然选择去追寻真理。

### 3. **《教父》**
  - **故事背景**：马里奥·普佐的小说讲述了一个黑手党家族的兴衰。
  - **矛盾冲突**：维托·柯里昂和迈克尔·柯里昂都在为家族而战，尽管他们所做的事常常是违法和道德上有争议的。迈克尔的选择从一开始是出于对家族的保护，但最终却使他变成了他曾经反对的人。故事探讨了权力、家庭和道德之间的复杂关系。

### 4. **《分裂》**
  - **故事背景**：这部电影探讨了一个人拥有多重人格的故事。
  - **矛盾冲突**：主角凯文有23个不同的人格，每个选择和行为都反映了他内心的斗争。尽管有些人格表现出善良和保护欲，但另一些人格则表现出极端的暴力和控制欲。最终，凯文的选择和行为导致了无法挽回的悲剧。

### 5. **《黑天鹅》**
  - **故事背景**：讲述了一名芭蕾舞者为了追求完美而走向疯狂的故事。
  - **矛盾冲突**：妮娜在追求艺术的过程中，不断压抑自己的真实自我，最终在完美与自我毁灭之间挣扎。她的选择看似是为了艺术的卓越，但却导致了她的精神崩溃和死亡。

### 深层思考

这些故事都展示了角色在道德、选择和后果之间的复杂交织。每个人的选择虽然在某种程度上是“正确”的，但从更宏观的视角来看，这些选择却加剧了整体的悲剧。这引发了对传统道德观念的质疑，特别是在现代社会中，随着环境和社会条件的变化，“好人”和“坏人”的界限变得越来越模糊。

### 结论

人们往往对那些充满矛盾和冲突的故事产生共鸣，因为这些故事反映了人类内心深处的复杂性和脆弱性。它们迫使我们反思道德标准、身份认同以及在面对困境时的选择。这种深刻的冲击感让我们意识到，生活并不像黑白分明的故事那样简单，而是充满了灰色地带和道德困境。
```

</details>

---

### Sample 9 — `general_knowledge`

**ID:** gp85 | **Sources:** 6 raw samples | **Padding:** 49 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> Can you write a code snippet which change the color of this SVG to another one?
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M7.89127 5.92513L5.96127 7.85513C5.85127 7.96513 5.68127 7.96513 5.57127 7.85513L3.68127 5.95513L6.00127 3.63513L7.89127 5.53513C8.00127 5.64513 8.00127 5.81513 7.89127 5.92513Z" fill="#E19747"/>
<path d="M9.47129 2.00512C9.20129 2.02512 9.00129 2.25512 9.00129 2.52512V5.18512C9.00129 5.43512 8.68129 5.52512 8.54129 5.32512L7.81129 4.20512C7.57129 3.84512 7.00129 4.01512 7.00129 4.44512V6.34512C7.00129 6.71512 6.71129 7.00512 6.35129 7.00512H5.06129C4.67129 7.00512 4.48129 7.47512 4.75129 7.75512L5.57129 8.58512C5.73129 8.73512 5.62129 9.00512 5.40129 9.00512H2.52129C2.25129 9.00512 2.02129 9.20512 2.00129 9.47512C1.99129 9.76512 2.22129 10.0051 2.50129 10.0051H7.37732L9.00129 9.00512L10.0013 7.36592V2.50512C10.0013 2.21512 9.76129 1.98512 9.47129 2.00512Z" fill="#FEEFC2"/>
<path d="M7.20131 10.0951L10.0713 7.21506C10.3713 6.91506 10.8613 6.92506 11.1813 7.24506C11.4913 7.55506 11.5013 8.05506 11.2013 8.35506L10.7879 10.9139L8.33131 11.2251C8.03131 11.5351 7.53131 11.5151 7.22131 11.2051C6.91131 10.8951 6.90131 10.3951 7.20131 10.0951Z" fill="#44911B"/>
<path d="M11.3524 15.7537C11.4872 18.9988 12.8273 22.081 15.1313 24.385L20.4713 29.725C20.8313 30.085 21.4213 30.085 21.7913 29.725L29.7313 21.765C30.0913 21.405 30.0913 20.805 29.7313 20.445L24.3913 15.105C22.086 12.7997 19.0015 11.4506 15.7627 11.316L15.6875 11.3788L12.3471 12.5174L11.4141 15.6796L11.3524 15.7537Z" fill="#44911B"/>
<path d="M6.89128 3.60512L3.60128 6.89512C3.44128 7.05512 3.19128 7.06512 3.03128 6.90512L2.11128 5.98512C1.96128 5.83512 1.96128 5.57512 2.12128 5.41512L5.40128 2.12512C5.56128 1.96512 5.81128 1.95512 5.97128 2.11512L6.90128 3.04512C7.05128 3.19512 7.05128 3.44512 6.89128 3.60512Z" fill="#F3C07B"/>
<path d="M8.24078 11.3045C8.27254 11.281 8.30283 11.2546 8.33132 11.2251L11.2013 8.35512C11.2478 8.30859 11.2869 8.25726 11.3186 8.20258L11.3413 8.22509L13.9213 10.7951C14.2413 11.1251 14.6913 11.3051 15.1513 11.3051C15.3556 11.3039 15.5594 11.3076 15.7627 11.316L15.6875 11.3789L15.2578 14.4727C15.1965 14.8914 14.9148 15.1926 14.4961 15.2539L11.4141 15.6797L11.3524 15.7537C11.3438 15.5481 11.3401 15.3419 11.3413 15.1351C11.3413 14.6651 11.1613 14.2251 10.8313 13.8951L8.25128 11.3151L8.24078 11.3045Z" fill="#F3C07B"/>
<path d="M20.9913 26.7751L26.7313 21.0251C26.8513 20.9051 26.8513 20.7051 26.7313 20.5851L23.4213 17.2651C23.3013 17.1451 23.1013 17.1451 22.9813 17.2651L17.2413 23.0151C17.1213 23.1351 17.1213 23.3351 17.2413 23.4551L20.5513 26.7751C20.6713 26.8951 20.8713 26.8951 20.9913 26.7751Z" fill="#F3C07B"/>
<path d="M12.6713 5.01509C12.6713 5.38512 12.3713 5.68509 12.0013 5.68509C11.6313 5.68509 11.3313 5.38512 11.3313 5.01509C11.3313 4.64506 11.6313 4.34509 12.0013 4.34509C12.3713 4.34509 12.6713 4.64506 12.6713 5.01509Z" fill="#FFDEA7"/>
<path d="M6.00128 12.0051C6.00128 12.5574 5.55357 13.0051 5.00128 13.0051C4.449 13.0051 4.00128 12.5574 4.00128 12.0051C4.00128 11.4528 4.449 11.0051 5.00128 11.0051C5.55357 11.0051 6.00128 11.4528 6.00128 12.0051Z" fill="#FFDEA7"/>
</svg>

<|assistant|> Sure, assuming you want to change only current filled colors with a new color, you can use JavaScript to replace attributes. First, specify the SVG in the HTML:

```html
<div id="mySVG">
    <!-- Your SVG code here -->
</div>
```

Then use the following JavaScript code:

```javascript
// Specify the new color here
let newColor = "#FF0000"; // New color is red

// Get SVG
let svg = document.getElementById('mySVG');

// Get all the paths inside the SVG
let paths = svg.getElementsByTagName('path');

// Change the fill color of each path
for (var i=0; i<paths.length; i++) {
    paths[i].setAttribute('fill', newColor);
}
```

This code will change fill color of all the `path` elements inside the SVG to the new color. Keep in mind that if not every `path` element should be changed, or if there is different colors in different `path` elements, you will need to add conditions to the above loop to only select `path` elements that have the correct original color.<|end_of_text|><|user|> If Martin consumes 1/2 cup of berries daily and the grocery store is selling packages of berries (1 cup per pack) for $2.00, what will be his total expenditure on berries in a period of 30 days?

<|assistant|> Martin consumes 1/2 cup of berries daily, so in 30 days, he will consume 1/2 x 30 = 15 cups of berries.
The grocery store is selling 1 cup of berries for $2.00, so Martin's total expenditure on berries in 30 days will be 15 x $2.00 = $30.00.
#### 30
The answer is: 30<|end_of_text|><|user|> A polynomial with integer coefficients is of the form
\[x^3 + a_2 x^2 + a_1 x - 11 = 0.\]Enter all the possible integer roots of this polynomial, separated by commas.

<|assistant|> By the Rational Root Theorem, any rational root of this polynomial must be a divisor of 11.
The divisors of 11 are $\pm 1$ and $\pm 11$, so the possible integer roots are $\boxed{-11, -1, 1, 11}$.
The answer is: -11,-1,1,11<|end_of_text|><|user|> You have been given a set of ten numbered boxes, each containing a different amount of gold coins. However, one of the boxes is rigged with a sensor that will trigger an alarm if the box is opened at a certain time. Your task is to find the box with the most coins by opening only two boxes. To complicate matters, you have to do this blindfolded.
How will you come up with a strategy to solve this puzzle without being able to see the boxes? Will you use your other senses or rely on probability? Consider edge cases and unexpected inputs that could throw off your approach. Can you come up with a foolproof plan that guarantees success?

<|assistant|> I'm happy to help you with this puzzle! To start, let's consider the information we have:

1. We have 10 numbered boxes, each containing a different amount of gold coins.
2. One of the boxes is rigged with a sensor that will trigger an alarm if the box is opened at a certain time.
3. We have to open only two boxes to find the box with the most coins.
4. We have to solve the puzzle blindfolded, so we can't see the boxes or their contents.

Now, let's think about how we can use our other senses to help us solve the puzzle. Since we can't see, we'll have to rely on our sense of touch and hearing. Here's a possible strategy:

1. First, we'll use our sense of touch to feel the weight of each box. We can assume that the box with the most coins will be the heaviest.
2. Next, we'll use our sense of hearing to listen for any sounds that might indicate which box has the sensor. When we open a box, we'll carefully listen for any beeping or other sounds that might indicate that we've triggered the alarm.
3. Now, let's consider probability. Since we don't know which box has the sensor, we'll have to assume that each box has an equal chance of having the sensor. Therefore, we'll open the first two boxes randomly.
4. Once we've opened the first two boxes, we'll use our sense of touch to feel the weight of the remaining boxes. Since we've already opened two boxes, we'll have a better idea of the weight distribution of the remaining boxes. We can then choose the heaviest box remaining to open next.
5. We'll continue this process of opening boxes, using our sense of touch and hearing to guide us, until we've found the box with the most coins.

Now, let's consider edge cases and unexpected inputs that could throw off our approach. What if the box with the sensor is the heaviest box? In that case, we might accidentally trigger the alarm when we open it. To mitigate this risk, we can modify our strategy as follows:

1. Instead of opening the heaviest box first, we'll open the lightest box first. This way, if the lightest box has the sensor, we'll trigger the alarm immediately, and we'll know that the heaviest box must be the one with the most coins.
2. If we don't trigger the alarm when we open the lightest box, we'll proceed with opening the next heaviest box. We'll continue this process until we've found the box with the most coins.

I believe this strategy is foolproof, as it takes into account all the available information and minimizes the risk of triggering the alarm. By relying on our sense of touch, hearing, and probability, we can solve the puzzle blindfolded and find the box with the most coins.<|end_of_text|><|user|> Solve the following math problem step-by-step.
Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
Find the sum of the rational roots of $g(x)=x^3-9x^2+16x-4$.

<|assistant|> Step 1:  The problem requires finding the sum of the rational roots of the given polynomial $g(x) = x^3 - 9x^2 + 16x - 4$.
Step 2:  To approach this, we can first apply the Rational Root Theorem to narrow down the possible rational roots of the polynomial.
Step 3:  According to the Rational Root Theorem, any rational root $p/q$ of $g(x)$ must have $p$ divide the constant term, which is 4, and $q$ divide the leading coefficient, which is 1.
Step 4:  This information tells us that any rational root must be an integer divisor of 4.
Step 5:  By systematically trying out the integer factors of 4 as potential roots, we find that $g(2) = 8 - 9*4 + 16*2 - 4 = 0$. Therefore, by the Factor theorem, we can deduce that $x-2$ is a factor of $g(x)$.
Step 6:  Using polynomial division, we can express $g(x)$ as $g(x) = (x-2)(x^2 - 7x + 2)$.
Step 7:  To determine the remaining roots of $g(x)$, we need to find the roots of the quadratic polynomial $x^2 - 7x + 2$ using the quadratic formula.
Step 8:  Applying the quadratic formula gives us $x = \frac{7 \pm \sqrt{49 - 8}}{2} = \frac{7 \pm \sqrt{41}}{2}$.
Step 9:  As these roots are not rational, the sum of the rational roots of $g(x)$ is $\boxed{2}$.

Answer:
\boxed{2}<|end_of_text|><|user|> could you render this code in lua for me?
def fibfib(n, m):
    if m > n or n < 0 or m < 0:
        raise ValueError("Invalid arguments passed.")

    # Initialize a list to store the sequence value at each step
    fibfib_seq = [0] * max(3, n+1)

    # Set base cases
    fibfib_seq[0] = 0
    fibfib_seq[1] = 0
    fibfib_seq[2] = 1

    # Build the sequence recursively
    for i in range(3, n+1):
        if i < m:
            fibfib_seq[i] = fibfib_seq[i-1] + fibfib_seq[i-2] + fibfib_seq[i-3]
        else:
            fibfib_seq[i] = fibfib_seq[i-1] + fibfib_seq[i-2] + fibfib_seq[i-3] - fibfib_seq[i-m]

    return fibfib_seq[n]

<|assistant|> Sure, here is the corresponding Lua code:

```lua
function fibfib(n, m)
    if m > n or n < 0 or m < 0 then
        error("Invalid arguments passed.")
    end

    -- Initialize a list to store the sequence value at each step
    local fibfib_seq = {}
    for i = 0, math.max(3, n) do
        fibfib_seq[i] = 0
    end

    -- Set base cases
    fibfib_seq[0] = 0
    fibfib_seq[1] = 0
    fibfib_seq[2] = 1

    -- Build the sequence recursively
    for i=3, n do
        if i < m then
            fibfib_seq[i] = fibfib_seq[i-1] + fibfib_seq[i-2] + fibfib_seq[i-3]
        else
            fibfib_seq[i] = fibfib_seq[i-1] + fibfib_seq[i-2] + fibfib_seq[i-3] - fibfib_seq[i-m]
        end
    end

    return fibfib_seq[n]
end
```
In Lua, arrays are indexed starting from 1, but in this code, I'm using it starting from 0 to match the Python code behaviour.
```

</details>

---

### Sample 10 — `math_competition_hard`

**ID:** gp471 | **Sources:** 6 raw samples | **Padding:** 318 tokens

<details>
<summary>Click to expand full sample</summary>

```
<|user|> Problem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number

$$
x^{p}+y^{p}+z^{p}-x-y-z
$$

is a product of exactly three distinct prime numbers.

<|assistant|> Solution. Let $A=x^{p}+y^{p}+z^{p}-x-y-z$. For $p=2$, we take $x=y=4$ and $z=3$. Then $A=30=2 \cdot 3 \cdot 5$. For $p=3$ we can take $x=3$ and $y=2$ and $z=1$. Then again $A=30=2 \cdot 3 \cdot 5$. For $p=5$ we can take $x=2$ and $y=1$ and $z=1$. Again $A=30=2 \cdot 3 \cdot 5$.

Assume now that $p \geqslant 7$. Working modulo 2 and modulo 3 we see that $A$ is divisible by both 2 and 3. Moreover, by Fermat's Little Theorem, we have

$$
x^{p}+y^{p}+z^{p}-x-y-z \equiv x+y+z-x-y-z=0 \bmod p \text {. }
$$

Therefore, by the given condition, we have to solve the equation

$$
x^{p}+y^{p}+z^{p}-x-y-z=6 p
$$

If one of the numbers $x, y$ and $z$ is bigger than or equal to 2 , let's say $x \geqslant 2$, then

$$
6 p \geqslant x^{p}-x=x\left(x^{p-1}-1\right) \geqslant 2\left(2^{p-1}-1\right)=2^{p}-2
$$

It is easy to check by induction that $2^{n}-2>6 n$ for all natural numbers $n \geqslant 6$. This contradiction shows that there are no more values of $p$ which satisfy the required property.

Remark. There are a couple of other ways to prove that $2^{p}-2>6 p$ for $p \geqslant 7$. For example, we can use the Binomial Theorem as follows:

$$
2^{p}-2 \geqslant 1+p+\frac{p(p-1)}{2}+\frac{p(p-1)(p-2)}{6}-2 \geqslant 1+p+3 p+5 p-2>6 p
$$

We can also use Bernoulli's Inequality as follows:

$$
2^{p}-2=8(1+1)^{p-3}-2 \geqslant 8(1+(p-3))-2=8 p-18>6 p
$$

The last inequality is true for $p \geqslant 11$. For $p=7$ we can see directly that $2^{p}-2>6 p$.

One can also use calculus to show that $f(x)=2^{x}-6 x$ is increasing for $x \geqslant 5$.<|end_of_text|><|user|> Problem 2. Let $a, b$ be two distinct real numbers and let $c$ be a positive real number such that

$$
a^{4}-2019 a=b^{4}-2019 b=c .
$$

Prove that $-\sqrt{c}<a b<0$.

<|assistant|> Solution. Firstly, we see that

$$
2019(a-b)=a^{4}-b^{4}=(a-b)(a+b)\left(a^{2}+b^{2}\right)
$$

Since $a \neq b$, we get $(a+b)\left(a^{2}+b^{2}\right)=2019$, so $a+b \neq 0$. Thus

$$
\begin{aligned}
2 c & =a^{4}-2019 a+b^{4}-2019 b \\
& =a^{4}+b^{4}-2019(a+b) \\
& =a^{4}+b^{4}-(a+b)^{2}\left(a^{2}+b^{2}\right) \\
& =-2 a b\left(a^{2}+a b+b^{2}\right)
\end{aligned}
$$

Hence $a b\left(a^{2}+a b+b^{2}\right)=-c0
$$

thus $a b-a b$ (the equality does not occur since $a+b \neq 0$ ). So

$$
-c=a b\left(a^{2}+a b+b^{2}\right)<-(a b)^{2} \Longrightarrow(a b)^{2}<c \Rightarrow-\sqrt{c}<a b<\sqrt{c}
$$

Therefore, we have $-\sqrt{c}<a b<0$.

Remark. We can get $c=-a b\left(a^{2}+a b+b^{2}\right)$ in several other ways. For example using that,

$$
(a-b) c=a\left(b^{4}-2019 b\right)-b\left(a^{4}-2019 a\right)=a b\left(b^{3}-a^{3}\right)=a b(b-a)\left(a^{2}+a b+b^{2}\right)
$$

We can also divide $f(x)=x^{4}-2019 x-c$ by $(x-a)(x-b)$ and look at the constant term of the remainder.<|end_of_text|><|user|> Problem 3. Triangle $A B C$ is such that $A B<A C$. The perpendicular bisector of side $B C$ intersects lines $A B$ and $A C$ at points $P$ and $Q$, respectively. Let $H$ be the orthocentre of triangle $A B C$, and let $M$ and $N$ be the midpoints of segments $B C$ and $P Q$, respectively. Prove that lines $H M$ and $A N$ meet on the circumcircle of $A B C$.

<|assistant|> Solution. We have

$$
\angle A P Q=\angle B P M=90^{\circ}-\angle M B P=90^{\circ}-\angle C B A=\angle H C B
$$

and

$$
\angle A Q P=\angle M Q C=90^{\circ}-\angle Q C M=90^{\circ}-\angle A C B=\angle C B H
$$

From these two equalities, we see that the triangles $A P Q$ and $H C B$ are similar. Moreover, since $M$ and $N$ are the midpoints of the segments $B C$ and $P Q$ respectively, then the triangles $A Q N$ and $H B M$ are also similar. Therefore, we have $\angle A N Q=\angle H M B$.

![](https://cdn.mathpix.com/cropped/2024_06_05_fe2687448771fc19bd4eg-4.jpg?height=883&width=545&top_left_y=865&top_left_x=779)

Let $L$ be the intersection of $A N$ and $H M$. We have

$$
\angle M L N=180^{\circ}-\angle L N M-\angle N M L=180^{\circ}-\angle L M B-\angle N M L=180^{\circ}-\angle N M B=90^{\circ} .
$$

Now let $D$ be the point on the circumcircle of $A B C$ diametrically oposite to $A$. It is known that $D$ is also the relfection of point $H$ over the point $M$. Therefore, we have that $D$ belongs on $M H$ and that $\angle D L A=\angle M L A=\angle M L N=90^{\circ}$. But, as $D A$ is the diameter of the circumcirle of $A B C$, the condition that $\angle D L A=90^{\circ}$ is enough to conclude that $L$ belongs on the circumcircle of $A B C$.

Remark. There is a spiral similarity mapping $A Q P$ to $H B C$. Since the similarity maps $A N$ to $H M$, it also maps $A H$ to $N M$, and since these two lines are parallel, the centre of the similarity is $L=A N \cap H M$. Since the similarity maps $B C$ to $Q P$, its centre belongs on the circumcircle of $B C X$, where $X=B Q \cap P C$. But $X$ is the reflection of $A$ on $Q M$ and so it must belong on the circumcircle of $A B C$. Hence so must $L$.

Remark. Students have also submitted correct proofs using radical axes, harmonic quadruples, coordinate geometry and complex numbers.<|end_of_text|><|user|> Problem 4. A $5 \times 100$ table is divided into 500 unit square cells, where $n$ of them are coloured black and the rest are coloured white. Two unit square cells are called adjacent if they share a common side. Each of the unit square cells has at most two adjacent black unit square cells. Find the largest possible value of $n$.

<|assistant|> Solution. If we colour all the cells along all edges of the board together with the entire middle row except the second and the last-but-one cell, the condition is satisfied and there are 302 black cells. The figure below exhibits this colouring for the $5 \times 8$ case.

![](https://cdn.mathpix.com/cropped/2024_06_05_fe2687448771fc19bd4eg-5.jpg?height=320&width=493&top_left_y=629&top_left_x=816)

We can cover the table by one fragment like the first one on the figure below, 24 fragments like the middle one, and one fragment like the third one.

![](https://cdn.mathpix.com/cropped/2024_06_05_fe2687448771fc19bd4eg-5.jpg?height=317&width=737&top_left_y=1132&top_left_x=694)

In each fragment, among the cells with the same letter, there are at most two coloured black, so the total number of coloured cells is at most $(5+24 \cdot 6+1) \cdot 2+2=302$.<|end_of_text|><|user|> Problem 1. A trapezoid $A B C D(A B \| C D, A B>C D)$ is circumscribed. The incircle of the triangle $A B C$ touches the lines $A B$ and $A C$ at the points $M$ and $N$, respectively. Prove that the incenter of the trapezoid $A B C D$ lies on the line $M N$.

![](https://cdn.mathpix.com/cropped/2024_06_05_bc5716af31e6d23d397cg-1.jpg?height=437&width=643&top_left_y=752&top_left_x=661)

<|assistant|> ## Solution.

Version 1. Let $I$ be the incenter of triangle $A B C$ and $R$ be the common point of the lines $B I$ and $M N$. Since

$$
m(\widehat{A N M})=90^{\circ}-\frac{1}{2} m(\widehat{M A N}) \quad \text { and } \quad m(\widehat{B I C})=90^{\circ}+\frac{1}{2} m(\widehat{M A N})
$$

the quadrilateral $I R N C$ is cyclic.

It follows that $m(\widehat{B R C})=90^{\circ}$ and therefore

$$
m(\widehat{B C R})=90^{\circ}-m(\widehat{C B R})=90^{\circ}-\frac{1}{2}\left(180^{\circ}-m(\widehat{B C D})\right)=\frac{1}{2} m(\widehat{B C D})
$$

So, $(C R$ is the angle bisector of $\widehat{D C B}$ and $R$ is the incenter of the trapezoid.

Version 2. If $R$ is the incentre of the trapezoid $A B C D$, then $B, I$ and $R$ are collinear,

and $m(\widehat{B R C})=90^{\circ}$.

The quadrilateral $I R N C$ is cyclic.

Then $m(\widehat{M N C})=90^{\circ}+\frac{1}{2} \cdot m(\widehat{B A C})$

and $m(\widehat{R N C})=m(\widehat{B I C})=90^{\circ}+\frac{1}{2} \cdot m(\widehat{B A C})$,
so that $m(\widehat{M N C})=m(\widehat{R N C})$ and the points $M, R$ and $N$ are collinear.

Version 3. If $R$ is the incentre of the trapezoid $A B C D$, let $M^{\prime} \in(A B)$ and $N^{\prime} \in(A C)$ be the unique points, such that $R \in M^{\prime} N^{\prime}$ and $\left(A M^{\prime}\right) \equiv\left(A N^{\prime}\right)$.

Let $S$ be the intersection point of $C R$ and $A B$. Then $C R=R S$.

Consider $K \in A C$ such that $S K \| M^{\prime} N^{\prime}$. Then $N^{\prime}$ is the midpoint of $(C K)$.

We deduce

$$
A N^{\prime}=\frac{A K+A C}{2}=\frac{A S+A C}{2}=\frac{A B-B S+A C}{2}=\frac{A B+A C-B C}{2}=A N
$$

We conclude that $N=N^{\prime}$, hence $M=M^{\prime}$, and $R, M, N$ are collinear.<|end_of_text|><|user|> Problem 2. Let $a, b$ and $c$ be positive real numbers. Prove that

$$
\frac{8}{(a+b)^{2}+4 a b c}+\frac{8}{(b+c)^{2}+4 a b c}+\frac{8}{(c+a)^{2}+4 a b c}+a^{2}+b^{2}+c^{2} \geq \frac{8}{a+3}+\frac{8}{b+3}+\frac{8}{c+3}
$$

<|assistant|> Solution. Since $2 a b \leq a^{2}+b^{2}$, it follows that $(a+b)^{2} \leq 2\left(a^{2}+b^{2}\right)$

and $4 a b c \leq 2 c\left(a^{2}+b^{2}\right)$, for any positive reals $a, b, c$.

Adding these inequalities, we find

$$
(a+b)^{2}+4 a b c \leq 2\left(a^{2}+b^{2}\right)(c+1)
$$

so that

$$
\frac{8}{(a+b)^{2}+4 a b c} \geq \frac{4}{\left(a^{2}+b^{2}\right)(c+1)}
$$

Using the AM-GM inequality, we have

$$
\frac{4}{\left(a^{2}+b^{2}\right)(c+1)}+\frac{a^{2}+b^{2}}{2} \geq 2 \sqrt{\frac{2}{c+1}}=\frac{4}{\sqrt{2(c+1)}}
$$

respectively

$$
\frac{c+3}{8}=\frac{(c+1)+2}{8} \geq \frac{\sqrt{2(c+1)}}{4}
$$

We conclude that

$$
\frac{4}{\left(a^{2}+b^{2}\right)(c+1)}+\frac{a^{2}+b^{2}}{2} \geq \frac{8}{c+3}
$$

and finally

$$
\frac{8}{(a+b)^{2}+4 a b c}+\frac{8}{(a+c)^{2}+4 a b c}+\frac{8}{(b+c)^{2}+4 a b c}+a^{2}+b^{2}+c^{2} \geq \frac{8}{a+3}+\frac{8}{b+3}+\frac{8}{c+3}
$$
```

</details>

---

## Notes

- Every sample is **exactly 4096 tokens** — no more, no less
- Short bins are padded with `<|pad|>` (token ID 130718) to reach 4096
- Samples within a bin are separated by `<|end_of_text|>` (token ID 130717)
- The `input_ids` field in the JSONL contains the exact 4096 token IDs (source of truth)
- Samples are grouped by domain first; cross-domain bins contain leftovers from multiple domains
