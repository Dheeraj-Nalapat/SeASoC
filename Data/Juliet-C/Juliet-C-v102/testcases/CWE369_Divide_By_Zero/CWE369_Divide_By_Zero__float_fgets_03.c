/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE369_Divide_By_Zero__float_fgets_03.c
Label Definition File: CWE369_Divide_By_Zero__float.label.xml
Template File: sources-sink-03.tmpl.c
*/
/*
 * @description
 * CWE: 369 Divide By Zero
 * BadSource: fgets Read data from the console using fgets()
 * GoodSource: Non-zero
 * Sink:
 *    BadSink : Divide a constant by data
 * Flow Variant: 03 Control flow: if(5==5) and if(5!=5)
 *
 * */

#include "std_testcase.h"

#define CHAR_ARRAY_SIZE sizeof(data)*sizeof(data)

#ifndef OMITBAD

void CWE369_Divide_By_Zero__float_fgets_03_bad()
{
    float data;
    /* Initialize data */
    data = 0.0F;
    if(5==5)
    {
        {
            char input_buf[CHAR_ARRAY_SIZE];
            /* POTENTIAL FLAW: Use a value input from the console using fgets() */
            fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
            /* Convert to float */
            data = (float)atof(input_buf);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a value not equal to zero */
        data = 7.0F;
    }
    /* POTENTIAL FLAW: Possibly divide by zero */
    printDoubleLine((double)(100.0 / data));
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the 5==5 to 5!=5 */
static void goodG2B1()
{
    float data;
    /* Initialize data */
    data = 0.0F;
    if(5!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char input_buf[CHAR_ARRAY_SIZE];
            /* POTENTIAL FLAW: Use a value input from the console using fgets() */
            fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
            /* Convert to float */
            data = (float)atof(input_buf);
        }
    }
    else
    {
        /* FIX: Use a value not equal to zero */
        data = 7.0F;
    }
    /* POTENTIAL FLAW: Possibly divide by zero */
    printDoubleLine((double)(100.0 / data));
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    float data;
    /* Initialize data */
    data = 0.0F;
    if(5==5)
    {
        /* FIX: Use a value not equal to zero */
        data = 7.0F;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char input_buf[CHAR_ARRAY_SIZE];
            /* POTENTIAL FLAW: Use a value input from the console using fgets() */
            fgets(input_buf, CHAR_ARRAY_SIZE, stdin);
            /* Convert to float */
            data = (float)atof(input_buf);
        }
    }
    /* POTENTIAL FLAW: Possibly divide by zero */
    printDoubleLine((double)(100.0 / data));
}

void CWE369_Divide_By_Zero__float_fgets_03_good()
{
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
    CWE369_Divide_By_Zero__float_fgets_03_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE369_Divide_By_Zero__float_fgets_03_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif