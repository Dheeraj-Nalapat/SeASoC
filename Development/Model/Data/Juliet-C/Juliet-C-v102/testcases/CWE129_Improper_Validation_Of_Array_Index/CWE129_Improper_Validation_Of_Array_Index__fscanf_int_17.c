/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__fscanf_int_17.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-17.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Larger than zero but less than 10
 * Sinks: int
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE129_Improper_Validation_Of_Array_Index__fscanf_int_17_bad()
{
    int h,i,j,k;
    int data;
    data = -1; /* Initialize data */
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    for(i = 0; i < 1; i++)
    {
        fscanf (stdin, "%d", &data);
    }
    for(j = 0; j < 1; j++)
    {
        {
            int data_buf[10] = {0,1,2,3,4,5,6,7,8,9};
            /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
             * This check does not check the upper bounds of the array index */
            if (data >= 0)
            {
                printIntLine(data_buf[data]);
            }
            else
            {
                printLine("ERROR: Array index is negative");
            }
        }
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int data_buf[10] = {0,1,2,3,4,5,6,7,8,9};
            /* FIX: Properly validate the array index */
            if (data >= 0 && data < 10)
            {
                printIntLine(data_buf[data]);
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

/* goodB2G() - use badsource and goodsink by changing the conditions on the third and fourth for statements */
static void goodB2G()
{
    int h,i,j,k;
    int data;
    data = -1; /* Initialize data */
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    for(i = 0; i < 1; i++)
    {
        fscanf (stdin, "%d", &data);
    }
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int data_buf[10] = {0,1,2,3,4,5,6,7,8,9};
            /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
             * This check does not check the upper bounds of the array index */
            if (data >= 0)
            {
                printIntLine(data_buf[data]);
            }
            else
            {
                printLine("ERROR: Array index is negative");
            }
        }
    }
    for(k = 0; k < 1; k++)
    {
        {
            int data_buf[10] = {0,1,2,3,4,5,6,7,8,9};
            /* FIX: Properly validate the array index */
            if (data >= 0 && data < 10)
            {
                printIntLine(data_buf[data]);
            }
            else
            {
                printLine("ERROR: Array index is out-of-bounds");
            }
        }
    }
}

/* goodG2B() - use goodsource and badsink by changing the conditions on the first and second for statements */
static void goodG2B()
{
    int h,i,j,k;
    int data;
    data = -1; /* Initialize data */
    for(h = 0; h < 1; h++)
    {
        /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
         * access an index of the array in the sink that is out-of-bounds */
        data = 7;
    }
    for(i = 0; i < 0; i++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        fscanf (stdin, "%d", &data);
    }
    for(j = 0; j < 1; j++)
    {
        {
            int data_buf[10] = {0,1,2,3,4,5,6,7,8,9};
            /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
             * This check does not check the upper bounds of the array index */
            if (data >= 0)
            {
                printIntLine(data_buf[data]);
            }
            else
            {
                printLine("ERROR: Array index is negative");
            }
        }
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int data_buf[10] = {0,1,2,3,4,5,6,7,8,9};
            /* FIX: Properly validate the array index */
            if (data >= 0 && data < 10)
            {
                printIntLine(data_buf[data]);
            }
            else
            {
                printLine("ERROR: Array index is out-of-bounds");
            }
        }
    }
}

void CWE129_Improper_Validation_Of_Array_Index__fscanf_int_17_good()
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
    CWE129_Improper_Validation_Of_Array_Index__fscanf_int_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE129_Improper_Validation_Of_Array_Index__fscanf_int_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
