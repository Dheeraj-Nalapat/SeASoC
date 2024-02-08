/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__rand_char_45.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-45.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: rand Set data to result of rand(), which may be zero
 * GoodSource: Larger than zero but less than 10
 * Sinks: char
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 45 Data flow: data passed as a static global variable from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

static int CWE129_Improper_Validation_Of_Array_Index__rand_char_45_bad_data;
static int CWE129_Improper_Validation_Of_Array_Index__rand_char_45_goodG2B_data;
static int CWE129_Improper_Validation_Of_Array_Index__rand_char_45_goodB2G_data;

#ifndef OMITBAD

static void bad_sink()
{
    int data = CWE129_Improper_Validation_Of_Array_Index__rand_char_45_bad_data;
    {
        char data_buf[10] = "AAAAAAAAA";
        /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
         * This check does not check the upper bounds of the array index */
        if (data >= 0)
        {
            printHexCharLine(data_buf[data]);
        }
        else
        {
            printLine("ERROR: Array index is negative");
        }
    }
}

void CWE129_Improper_Validation_Of_Array_Index__rand_char_45_bad()
{
    int data;
    data = -1; /* Initialize data */
    data = RAND32();
    CWE129_Improper_Validation_Of_Array_Index__rand_char_45_bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink()
{
    int data = CWE129_Improper_Validation_Of_Array_Index__rand_char_45_goodG2B_data;
    {
        char data_buf[10] = "AAAAAAAAA";
        /* POTENTIAL FLAW: Attempt to access an index of the array that is possibly out-of-bounds
         * This check does not check the upper bounds of the array index */
        if (data >= 0)
        {
            printHexCharLine(data_buf[data]);
        }
        else
        {
            printLine("ERROR: Array index is negative");
        }
    }
}

static void goodG2B()
{
    int data;
    data = -1; /* Initialize data */
    /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
     * access an index of the array in the sink that is out-of-bounds */
    data = 7;
    CWE129_Improper_Validation_Of_Array_Index__rand_char_45_goodG2B_data = data;
    goodG2B_sink();
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G_sink()
{
    int data = CWE129_Improper_Validation_Of_Array_Index__rand_char_45_goodB2G_data;
    {
        char data_buf[10] = "AAAAAAAAA";
        /* FIX: Properly validate the array index */
        if (data >= 0 && data < (10-1))  /* Include the -1 because we don't want to print null */
        {
            printHexCharLine(data_buf[data]);
        }
        else
        {
            printLine("ERROR: Array index is out-of-bounds");
        }
    }
}

static void goodB2G()
{
    int data;
    data = -1; /* Initialize data */
    data = RAND32();
    CWE129_Improper_Validation_Of_Array_Index__rand_char_45_goodB2G_data = data;
    goodB2G_sink();
}

void CWE129_Improper_Validation_Of_Array_Index__rand_char_45_good()
{
    goodG2B();
    goodB2G();
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
    CWE129_Improper_Validation_Of_Array_Index__rand_char_45_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE129_Improper_Validation_Of_Array_Index__rand_char_45_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
