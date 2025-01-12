#ifndef IDSP_ADDON_HPP
#define IDSP_ADDON_HPP

#include <array>
#include <initializer_list>
#include <type_traits>
#include <stdexcept> // Added for std::out_of_range

namespace idsp
{
    template <typename T, size_t N>
class c_vector { 
public:
    c_vector():
        data(),
        maximum_size(N),
        current_size(N)
    {};
    c_vector(std::initializer_list<T> init_list) : 
        data(), 
        maximum_size(N), 
        current_size(0)
    {
        if(init_list.size() > N)
            throw std::out_of_range("Initializer list size exceeds maximum size");
        for(const auto& value : init_list)
            data[current_size++] = value;
    }
    ~c_vector() = default;

    inline void push_back(const T& value) 
    { 
        if(current_size >= maximum_size) 
            throw std::out_of_range("push_back: c_vector is full");
        data[current_size++] = value; 
    }

    inline void pop_back() 
    { 
        if(current_size == 0) 
            throw std::out_of_range("pop_back: c_vector is empty");
        --current_size; 
    }

    inline size_t size() const { return current_size; }
    inline size_t capacity() const { return maximum_size; }

    inline bool empty() const { return current_size == 0; }

    inline const T& operator[](size_t index) const 
    { 
        if(index >= current_size) 
            throw std::out_of_range("Operator[]: Index out of range");
        return data[index]; 
    }

    inline T& operator[](const size_t index) 
    { 
        size_t i = maximum_size;
        if(index >= current_size) 
            throw std::out_of_range("Operator[]: Index out of range");
        return data[index]; 
    }

    inline T& at(const size_t index) 
    { 
        if(index >= current_size) 
            throw std::out_of_range("at: Index out of range");
        return data[index]; 
    }

    inline const T& at(const size_t index) const 
    { 
        if(index >= current_size) 
            throw std::out_of_range("at: Index out of range");
        return data[index]; 
    }

    inline void clear() { current_size = 0; }

    inline void assign(size_t count, const T& value)
    {
        if(count > maximum_size)
            throw std::out_of_range("assign: count exceeds maximum size");
        clear();
        for(size_t i = 0; i < count; ++i)
            data[current_size++] = value;
    }


    inline void resize(const size_t new_size)
    {
        if(new_size > maximum_size)
            throw std::out_of_range("resize: new size exceeds maximum size");
        current_size = new_size;
    }

    // inline void resize(const size_t new_size, const T& value = T())
    // {
    //     if(new_size > maximum_size)
    //         throw std::out_of_range("resize: new size exceeds maximum size");
    //     if(new_size > current_size)
    //     {
    //         for(size_t i = current_size; i < new_size; ++i)
    //             data[i] = value;
    //     }
    //     current_size = new_size;
    // }

    inline void fill(const double val)
    {
        if(std::is_arithmetic<T>::value)
        for(size_t i = 0; i < maximum_size; i++)
            data[i] = val;
    }

    inline T* begin() { return data.data(); }
    inline const T* begin() const { return data.data(); }
    inline T* end() { return data.data() + current_size; }
    inline const T* end() const { return data.data() + current_size; }


private:
    size_t maximum_size = N;
    size_t current_size = 0;
    std::array<T, N> data;
};
}

#endif // IDSP_ADDON_HPP    