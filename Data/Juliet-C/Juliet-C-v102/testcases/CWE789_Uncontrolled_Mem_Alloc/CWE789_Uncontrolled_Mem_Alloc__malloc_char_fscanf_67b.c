/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67b.c
Label Definition File: CWE789_Uncontrolled_Mem_Alloc__malloc.label.xml
Template File: sources-sinks-67b.tmpl.c
*/
/*
 * @description
 * CWE: 789 Uncontrolled Memory Allocation
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Small number greater than zero
 * Sinks:
 *    GoodSink: Allocate memory with malloc() and check the size of the memory to be allocated
 *    BadSink : Allocate memory with malloc(), but incorrectly check the size of the memory to be allocated
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#define HELLO_STRING "hello"

typedef struct _CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67_struct_type
{
    int a;
} CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67_struct_type;

#ifndef OMITBAD

void CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67b_bad_sink(CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67_struct_type my_struct)
{
    int data = my_struct.a;
    {
        char * char_string;
        /* FLAW: No MAXIMUM limitation for memory allocation and no check to ensure data is large enough
         * for the strcpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if (data > 0)
        {
            char_string = (char *)malloc(data*sizeof(char));
            /* Copy a small string into char_string$ */
            strcpy(char_string, HELLO_STRING);
            printLine(char_string);
            free(char_string);
        }
        else
        {
            printLine("Input is less than 0");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67b_goodG2B_sink(CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67_struct_type my_struct)
{
    int data = my_struct.a;
    {
        char * char_string;
        /* FLAW: No MAXIMUM limitation for memory allocation and no check to ensure data is large enough
         * for the strcpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if (data > 0)
        {
            char_string = (char *)malloc(data*sizeof(char));
            /* Copy a small string into char_string$ */
            strcpy(char_string, HELLO_STRING);
            printLine(char_string);
            free(char_string);
        }
        else
        {
            printLine("Input is less than 0");
        }
    }
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67b_goodB2G_sink(CWE789_Uncontrolled_Mem_Alloc__malloc_char_fscanf_67_struct_type my_struct)
{
    int data = my_struct.a;
    {
        char * char_string;
        /* FIX: Include a MAXIMUM limitation for memory allocation and a check to ensure data is large enough
         * for the strcpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if ((size_t)data > strlen(HELLO_STRING) && data < 100)
        {
            char_string = (char *)malloc(data*sizeof(char));
            /* Copy a small string into char_string$ */
            strcpy(char_string, HELLO_STRING);
            printLine(char_string);
            free(char_string);
        }
        else
        {
            printLine("Input is less than the length of the source string or too large");
        }
    }
}

#endif /* OMITGOOD */
