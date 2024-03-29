/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE197_Numeric_Truncation_Error__short_fgets_66a.c
Label Definition File: CWE197_Numeric_Truncation_Error__short.label.xml
Template File: sources-sink-66a.tmpl.c
*/
/*
 * @description
 * CWE: 197 Numeric Truncation Error
 * BadSource: fgets Read data from the console using fgets()
 * GoodSource: Less than CHAR_MAX
 * Sinks:
 *    BadSink : Convert data to a char
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

/* Must be at least 8 for atoi() to work properly */
#define CHAR_ARRAY_SIZE 8

#ifndef OMITBAD

/* bad function declaration */
void CWE197_Numeric_Truncation_Error__short_fgets_66b_bad_sink(short data_array[]);

void CWE197_Numeric_Truncation_Error__short_fgets_66_bad()
{
    short data;
    short data_array[5];
    /* Initialize data */
    data = -1;
    {
        char input_buf[CHAR_ARRAY_SIZE] = "";
        /* POTENTIAL FLAW: Use a number input from the console using fgets() */
        fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
        /* Convert to short */
        data = (short)atoi(input_buf);
    }
    /* put data in array */
    data_array[2] = data;
    CWE197_Numeric_Truncation_Error__short_fgets_66b_bad_sink(data_array);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE197_Numeric_Truncation_Error__short_fgets_66b_goodG2B_sink(short data_array[]);

static void goodG2B()
{
    short data;
    short data_array[5];
    /* Initialize data */
    data = -1;
    /* FIX: Use a positive integer less than CHAR_MAX*/
    data = CHAR_MAX-5;
    data_array[2] = data;
    CWE197_Numeric_Truncation_Error__short_fgets_66b_goodG2B_sink(data_array);
}

void CWE197_Numeric_Truncation_Error__short_fgets_66_good()
{
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
    CWE197_Numeric_Truncation_Error__short_fgets_66_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE197_Numeric_Truncation_Error__short_fgets_66_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
