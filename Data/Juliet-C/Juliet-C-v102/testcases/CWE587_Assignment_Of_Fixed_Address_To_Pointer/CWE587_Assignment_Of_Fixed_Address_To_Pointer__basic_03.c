/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03.c
Label Definition File: CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic.label.xml
Template File: point-flaw-03.tmpl.c
*/
/*
 * @description
 * CWE: 587 Assignment of Fixed Address to Pointer
 * Sinks:
 *    GoodSink: Assign the pointer to a stack variable
 *    BadSink : Assign fixed address to pointer
 * Flow Variant: 03 Control flow: if(5==5) and if(5!=5)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_bad()
{
    if(5==5)
    {
        {
            /* FLAW: Assigning fixed address to pointer */
            char *p = (char*)0x400000;
            printHexCharLine(*p);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Assign the pointer to a stack variable */
            char c = 'a';
            char *p = &c;
            printHexCharLine(*p);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(5!=5) instead of if(5==5) */
static void good1()
{
    if(5!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Assigning fixed address to pointer */
            char *p = (char*)0x400000;
            printHexCharLine(*p);
        }
    }
    else
    {
        {
            /* FIX: Assign the pointer to a stack variable */
            char c = 'a';
            char *p = &c;
            printHexCharLine(*p);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(5==5)
    {
        {
            /* FIX: Assign the pointer to a stack variable */
            char c = 'a';
            char *p = &c;
            printHexCharLine(*p);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Assigning fixed address to pointer */
            char *p = (char*)0x400000;
            printHexCharLine(*p);
        }
    }
}

void CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_good()
{
    good1();
    good2();
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
    CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
