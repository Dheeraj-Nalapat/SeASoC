/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__large_char_66a.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-66a.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: large Large index value that is greater than 10-1
 * GoodSource: Larger than zero but less than 10
 * Sinks: char
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE129_Improper_Validation_Of_Array_Index__large_char_66b_bad_sink(int data_array[]);

void CWE129_Improper_Validation_Of_Array_Index__large_char_66_bad()
{
    int data;
    int data_array[5];
    data = -1; /* Initialize data */
    /* FLAW: Use an invalid index */
    data = 10;
    /* put data in array */
    data_array[2] = data;
    CWE129_Improper_Validation_Of_Array_Index__large_char_66b_bad_sink(data_array);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE129_Improper_Validation_Of_Array_Index__large_char_66b_goodG2B_sink(int data_array[]);

static void goodG2B()
{
    int data;
    int data_array[5];
    data = -1; /* Initialize data */
    /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
     * access an index of the array in the sink that is out-of-bounds */
    data = 7;
    data_array[2] = data;
    CWE129_Improper_Validation_Of_Array_Index__large_char_66b_goodG2B_sink(data_array);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE129_Improper_Validation_Of_Array_Index__large_char_66b_goodB2G_sink(int data_array[]);

static void goodB2G()
{
    int data;
    int data_array[5];
    data = -1; /* Initialize data */
    /* FLAW: Use an invalid index */
    data = 10;
    data_array[2] = data;
    CWE129_Improper_Validation_Of_Array_Index__large_char_66b_goodB2G_sink(data_array);
}

void CWE129_Improper_Validation_Of_Array_Index__large_char_66_good()
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
    CWE129_Improper_Validation_Of_Array_Index__large_char_66_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE129_Improper_Validation_Of_Array_Index__large_char_66_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
