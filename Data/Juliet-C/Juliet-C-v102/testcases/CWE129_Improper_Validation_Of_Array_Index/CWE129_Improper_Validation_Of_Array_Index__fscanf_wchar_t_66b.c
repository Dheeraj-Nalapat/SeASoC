/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__fscanf_wchar_t_66b.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-66b.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Larger than zero but less than 10
 * Sinks: wchar_t
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE129_Improper_Validation_Of_Array_Index__fscanf_wchar_t_66b_bad_sink(int data_array[])
{
    /* copy data out of data_array */
    int data = data_array[2];
    {
        wchar_t data_buf[10] = L"AAAAAAAAA";
        /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
         * This check does not check the upper bounds of the array index */
        if (data >= 0)
        {
            printWcharLine(data_buf[data]);
        }
        else
        {
            printLine("ERROR: Array index is negative");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE129_Improper_Validation_Of_Array_Index__fscanf_wchar_t_66b_goodG2B_sink(int data_array[])
{
    int data = data_array[2];
    {
        wchar_t data_buf[10] = L"AAAAAAAAA";
        /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
         * This check does not check the upper bounds of the array index */
        if (data >= 0)
        {
            printWcharLine(data_buf[data]);
        }
        else
        {
            printLine("ERROR: Array index is negative");
        }
    }
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE129_Improper_Validation_Of_Array_Index__fscanf_wchar_t_66b_goodB2G_sink(int data_array[])
{
    int data = data_array[2];
    {
        wchar_t data_buf[10] = L"AAAAAAAAA";
        /* FIX: Properly validate the array index */
        if (data >= 0 && data < (10-1))  /* Include the -1 because we don't want to print null */
        {
            printWcharLine(data_buf[data]);
        }
        else
        {
            printLine("ERROR: Array index is out-of-bounds");
        }
    }
}

#endif /* OMITGOOD */
