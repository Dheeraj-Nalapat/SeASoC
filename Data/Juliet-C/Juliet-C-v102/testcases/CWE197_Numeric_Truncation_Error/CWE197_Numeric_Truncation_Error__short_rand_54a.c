/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE197_Numeric_Truncation_Error__short_rand_54a.c
Label Definition File: CWE197_Numeric_Truncation_Error__short.label.xml
Template File: sources-sink-54a.tmpl.c
*/
/*
 * @description
 * CWE: 197 Numeric Truncation Error
 * BadSource: rand Set data to result of rand(), which could be negative
 * GoodSource: Less than CHAR_MAX
 * Sink:
 *    BadSink : Convert data to a char
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE197_Numeric_Truncation_Error__short_rand_54b_bad_sink(short data);

void CWE197_Numeric_Truncation_Error__short_rand_54_bad()
{
    short data;
    /* Initialize data */
    data = -1;
    /* POTENTIAL FLAW: Use a random number */
    data = (short)RAND32();
    CWE197_Numeric_Truncation_Error__short_rand_54b_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE197_Numeric_Truncation_Error__short_rand_54b_goodG2B_sink(short data);

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    short data;
    /* Initialize data */
    data = -1;
    /* FIX: Use a positive integer less than CHAR_MAX*/
    data = CHAR_MAX-5;
    CWE197_Numeric_Truncation_Error__short_rand_54b_goodG2B_sink(data);
}

void CWE197_Numeric_Truncation_Error__short_rand_54_good()
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
    CWE197_Numeric_Truncation_Error__short_rand_54_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE197_Numeric_Truncation_Error__short_rand_54_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
