/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__large_wchar_t_04.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-04.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: large Large index value that is greater than 10-1
 * GoodSource: Larger than zero but less than 10
 * Sinks: wchar_t
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 04 Control flow: if(static_const_t) and if(static_const_f)
 *
 * */

#include "std_testcase.h"

/* The two variables below are declared "const", so a tool should
   be able to identify that reads of these will always return their
   initialized values. */
static const int static_const_t = 1; /* true */
static const int static_const_f = 0; /* false */

#ifndef OMITBAD

void CWE129_Improper_Validation_Of_Array_Index__large_wchar_t_04_bad()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        /* FLAW: Use an invalid index */
        data = 10;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    if(static_const_t)
    {
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
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
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
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_const_t to static_const_f */
static void goodB2G1()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        /* FLAW: Use an invalid index */
        data = 10;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
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
    else
    {
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
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        /* FLAW: Use an invalid index */
        data = 10;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    if(static_const_t)
    {
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
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
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
}

/* goodG2B1() - use goodsource and badsink by changing the first static_const_t to static_const_f */
static void goodG2B1()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Use an invalid index */
        data = 10;
    }
    else
    {
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    if(static_const_t)
    {
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
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
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
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Use an invalid index */
        data = 10;
    }
    if(static_const_t)
    {
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
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
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
}

void CWE129_Improper_Validation_Of_Array_Index__large_wchar_t_04_good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
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
    CWE129_Improper_Validation_Of_Array_Index__large_wchar_t_04_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE129_Improper_Validation_Of_Array_Index__large_wchar_t_04_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
