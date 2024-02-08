/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE197_Numeric_Truncation_Error__int_fgets_to_short_32.c
Label Definition File: CWE197_Numeric_Truncation_Error__int.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 197 Numeric Truncation Error
 * BadSource: fgets Read data from the console using fgets()
 * GoodSource: Less than CHAR_MAX
 * Sink: to_short
 *    BadSink : Convert data to a short
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#define CHAR_ARRAY_SIZE sizeof(data)*sizeof(data)

#ifndef OMITBAD

void CWE197_Numeric_Truncation_Error__int_fgets_to_short_32_bad()
{
    int data;
    int *data_ptr1 = &data;
    int *data_ptr2 = &data;
    /* Initialize data */
    data = -1;
    {
        int data = *data_ptr1;
        {
            char input_buf[CHAR_ARRAY_SIZE] = "";
            fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
            /* Convert to int */
            data = atoi(input_buf);
        }
        *data_ptr1 = data;
    }
    {
        int data = *data_ptr2;
        {
            /* POTENTIAL FLAW: Convert data to a short, possibly causing a truncation error */
            short s = (short)data;
            printf("%hd\n", s);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    int data;
    int *data_ptr1 = &data;
    int *data_ptr2 = &data;
    /* Initialize data */
    data = -1;
    {
        int data = *data_ptr1;
        /* FIX: Use a positive integer less than CHAR_MAX*/
        data = CHAR_MAX-5;
        *data_ptr1 = data;
    }
    {
        int data = *data_ptr2;
        {
            /* POTENTIAL FLAW: Convert data to a short, possibly causing a truncation error */
            short s = (short)data;
            printf("%hd\n", s);
        }
    }
}

void CWE197_Numeric_Truncation_Error__int_fgets_to_short_32_good()
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
    CWE197_Numeric_Truncation_Error__int_fgets_to_short_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE197_Numeric_Truncation_Error__int_fgets_to_short_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
