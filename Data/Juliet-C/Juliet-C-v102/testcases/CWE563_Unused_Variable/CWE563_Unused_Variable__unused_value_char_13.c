/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE563_Unused_Variable__unused_value_char_13.c
Label Definition File: CWE563_Unused_Variable__unused_value.label.xml
Template File: sources-sinks-13.tmpl.c
*/
/*
 * @description
 * CWE: 563 Unused Variable
 * BadSource:  Initialize data
 * GoodSource: Initialize and use data
 * Sinks:
 *    GoodSink: Use data
 *    BadSink : Initialize and use data
 * Flow Variant: 13 Control flow: if(global_const_five==5) and if(global_const_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE563_Unused_Variable__unused_value_char_13_bad()
{
    char data;
    if(global_const_five==5)
    {
        /* POTENTIAL FLAW: Initialize, but do not use data */
        data = 'C';
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Initialize and use data before it is overwritten */
        data = 'C';
        printHexCharLine(data);
    }
    if(global_const_five==5)
    {
        /* POTENTIAL FLAW: Possibly over-write the initial value of data before using it */
        data = 'Z';
        printHexCharLine(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use data without over-writing its value */
        printHexCharLine(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second global_const_five==5 to global_const_five!=5 */
static void goodB2G1()
{
    char data;
    if(global_const_five==5)
    {
        /* POTENTIAL FLAW: Initialize, but do not use data */
        data = 'C';
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Initialize and use data before it is overwritten */
        data = 'C';
        printHexCharLine(data);
    }
    if(global_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Possibly over-write the initial value of data before using it */
        data = 'Z';
        printHexCharLine(data);
    }
    else
    {
        /* FIX: Use data without over-writing its value */
        printHexCharLine(data);
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    char data;
    if(global_const_five==5)
    {
        /* POTENTIAL FLAW: Initialize, but do not use data */
        data = 'C';
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Initialize and use data before it is overwritten */
        data = 'C';
        printHexCharLine(data);
    }
    if(global_const_five==5)
    {
        /* FIX: Use data without over-writing its value */
        printHexCharLine(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Possibly over-write the initial value of data before using it */
        data = 'Z';
        printHexCharLine(data);
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first global_const_five==5 to global_const_five!=5 */
static void goodG2B1()
{
    char data;
    if(global_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Initialize, but do not use data */
        data = 'C';
    }
    else
    {
        /* FIX: Initialize and use data before it is overwritten */
        data = 'C';
        printHexCharLine(data);
    }
    if(global_const_five==5)
    {
        /* POTENTIAL FLAW: Possibly over-write the initial value of data before using it */
        data = 'Z';
        printHexCharLine(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use data without over-writing its value */
        printHexCharLine(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    char data;
    if(global_const_five==5)
    {
        /* FIX: Initialize and use data before it is overwritten */
        data = 'C';
        printHexCharLine(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Initialize, but do not use data */
        data = 'C';
    }
    if(global_const_five==5)
    {
        /* POTENTIAL FLAW: Possibly over-write the initial value of data before using it */
        data = 'Z';
        printHexCharLine(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use data without over-writing its value */
        printHexCharLine(data);
    }
}

void CWE563_Unused_Variable__unused_value_char_13_good()
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
    CWE563_Unused_Variable__unused_value_char_13_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE563_Unused_Variable__unused_value_char_13_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
