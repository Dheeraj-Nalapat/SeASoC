/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE789_Uncontrolled_Mem_Alloc__malloc_wchar_t_fgets_19.c
Label Definition File: CWE789_Uncontrolled_Mem_Alloc__malloc.label.xml
Template File: sources-sinks-19.tmpl.c
*/
/*
 * @description
 * CWE: 789 Uncontrolled Memory Allocation
 * BadSource: fgets Read data from the console using fgets()
 * GoodSource: Small number greater than zero
 * Sinks:
 *    GoodSink: Allocate memory with malloc() and check the size of the memory to be allocated
 *    BadSink : Allocate memory with malloc(), but incorrectly check the size of the memory to be allocated
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#define CHAR_ARRAY_SIZE sizeof(data)*sizeof(data)

#define HELLO_STRING L"hello"

#ifndef OMITBAD

void CWE789_Uncontrolled_Mem_Alloc__malloc_wchar_t_fgets_19_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    {
        char input_buf[CHAR_ARRAY_SIZE] = "";
        fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
        /* Convert to int */
        data = atoi(input_buf);
    }
    {
        wchar_t * wchar_t_string;
        /* FLAW: No MAXIMUM limitation for memory allocation and no check to ensure data is large enough
         * for the wcscpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if (data > 0)
        {
            wchar_t_string = (wchar_t *)malloc(data*sizeof(wchar_t));
            /* Copy a small string into wchar_t_string$ */
            wcscpy(wchar_t_string, HELLO_STRING);
            printWLine(wchar_t_string);
            free(wchar_t_string);
        }
        else
        {
            printLine("Input is less than 0");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        wchar_t * wchar_t_string;
        /* FIX: Include a MAXIMUM limitation for memory allocation and a check to ensure data is large enough
         * for the wcscpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if ((size_t)data > wcslen(HELLO_STRING) && data < 100)
        {
            wchar_t_string = (wchar_t *)malloc(data*sizeof(wchar_t));
            /* Copy a small string into wchar_t_string$ */
            wcscpy(wchar_t_string, HELLO_STRING);
            printWLine(wchar_t_string);
            free(wchar_t_string);
        }
        else
        {
            printLine("Input is less than the length of the source string or too large");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by switching statements around return */
static void goodB2G()
{
    int data;
    /* Initialize data */
    data = -1;
    {
        char input_buf[CHAR_ARRAY_SIZE] = "";
        fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
        /* Convert to int */
        data = atoi(input_buf);
    }
    {
        wchar_t * wchar_t_string;
        /* FIX: Include a MAXIMUM limitation for memory allocation and a check to ensure data is large enough
         * for the wcscpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if ((size_t)data > wcslen(HELLO_STRING) && data < 100)
        {
            wchar_t_string = (wchar_t *)malloc(data*sizeof(wchar_t));
            /* Copy a small string into wchar_t_string$ */
            wcscpy(wchar_t_string, HELLO_STRING);
            printWLine(wchar_t_string);
            free(wchar_t_string);
        }
        else
        {
            printLine("Input is less than the length of the source string or too large");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        wchar_t * wchar_t_string;
        /* FLAW: No MAXIMUM limitation for memory allocation and no check to ensure data is large enough
         * for the wcscpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if (data > 0)
        {
            wchar_t_string = (wchar_t *)malloc(data*sizeof(wchar_t));
            /* Copy a small string into wchar_t_string$ */
            wcscpy(wchar_t_string, HELLO_STRING);
            printWLine(wchar_t_string);
            free(wchar_t_string);
        }
        else
        {
            printLine("Input is less than 0");
        }
    }
}

/* goodG2B() - use goodsource and badsink by moving BadSource and BadSink to after return */
static void goodG2B()
{
    int data;
    /* Initialize data */
    data = -1;
    /* FIX: Use a relatively small number for memory allocation */
    data = 20;
    {
        wchar_t * wchar_t_string;
        /* FLAW: No MAXIMUM limitation for memory allocation and no check to ensure data is large enough
         * for the wcscpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if (data > 0)
        {
            wchar_t_string = (wchar_t *)malloc(data*sizeof(wchar_t));
            /* Copy a small string into wchar_t_string$ */
            wcscpy(wchar_t_string, HELLO_STRING);
            printWLine(wchar_t_string);
            free(wchar_t_string);
        }
        else
        {
            printLine("Input is less than 0");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        char input_buf[CHAR_ARRAY_SIZE] = "";
        fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
        /* Convert to int */
        data = atoi(input_buf);
    }
    {
        wchar_t * wchar_t_string;
        /* FLAW: No MAXIMUM limitation for memory allocation and no check to ensure data is large enough
         * for the wcscpy() function to not cause a buffer overflow */
        /* INCIDENTAL FLAW: The source could cause a type overrun in data or in the memory allocation */
        if (data > 0)
        {
            wchar_t_string = (wchar_t *)malloc(data*sizeof(wchar_t));
            /* Copy a small string into wchar_t_string$ */
            wcscpy(wchar_t_string, HELLO_STRING);
            printWLine(wchar_t_string);
            free(wchar_t_string);
        }
        else
        {
            printLine("Input is less than 0");
        }
    }
}

void CWE789_Uncontrolled_Mem_Alloc__malloc_wchar_t_fgets_19_good()
{
    goodB2G();
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE789_Uncontrolled_Mem_Alloc__malloc_wchar_t_fgets_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE789_Uncontrolled_Mem_Alloc__malloc_wchar_t_fgets_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
