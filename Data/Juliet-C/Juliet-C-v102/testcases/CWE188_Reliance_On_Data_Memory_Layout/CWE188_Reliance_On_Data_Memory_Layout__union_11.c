/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE188_Reliance_On_Data_Memory_Layout__union_11.c
Label Definition File: CWE188_Reliance_On_Data_Memory_Layout.label.xml
Template File: point-flaw-11.tmpl.c
*/
/*
 * @description
 * CWE: 188 Reliance on Data Memory Layout
 * Sinks: union
 *    GoodSink: Use mechanisms that are not dependent upon size, alignment/packing, or byte order to accomplish equivalent functionality
 *    BadSink : Modify a union member by relying on the byte-order, size, alignment/packing of struct and union fields
 * Flow Variant: 11 Control flow: if(global_returns_t()) and if(global_returns_f())
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE188_Reliance_On_Data_Memory_Layout__union_11_bad()
{
    if(global_returns_t())
    {
        {
            union
            {
                struct
                {
                    char a, b, c, d;
                } s;
                long l;
            } u;
            u.l = 0x10203040;
            /* FLAW: this operation depends on the byte-order, size, alignment/packing of
             * struct and union fields */
            u.s.d |= 0x80; /* "set the MSB" */
            printIntLine(u.l);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            long l;
            l = 0x10203040;
            /* FIX: Use mechanisms that are not dependent upon size, alignment/packing,
             * or byte order to accomplish equivalent functionality
             *
             * Technically this 'fix' here still depends on twos-complement hardware,
             * but everything nowadays is twos-complement
             */
            l |= LONG_MIN;
            printIntLine(l);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_returns_f()) instead of if(global_returns_t()) */
static void good1()
{
    if(global_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            union
            {
                struct
                {
                    char a, b, c, d;
                } s;
                long l;
            } u;
            u.l = 0x10203040;
            /* FLAW: this operation depends on the byte-order, size, alignment/packing of
             * struct and union fields */
            u.s.d |= 0x80; /* "set the MSB" */
            printIntLine(u.l);
        }
    }
    else
    {
        {
            long l;
            l = 0x10203040;
            /* FIX: Use mechanisms that are not dependent upon size, alignment/packing,
             * or byte order to accomplish equivalent functionality
             *
             * Technically this 'fix' here still depends on twos-complement hardware,
             * but everything nowadays is twos-complement
             */
            l |= LONG_MIN;
            printIntLine(l);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_returns_t())
    {
        {
            long l;
            l = 0x10203040;
            /* FIX: Use mechanisms that are not dependent upon size, alignment/packing,
             * or byte order to accomplish equivalent functionality
             *
             * Technically this 'fix' here still depends on twos-complement hardware,
             * but everything nowadays is twos-complement
             */
            l |= LONG_MIN;
            printIntLine(l);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            union
            {
                struct
                {
                    char a, b, c, d;
                } s;
                long l;
            } u;
            u.l = 0x10203040;
            /* FLAW: this operation depends on the byte-order, size, alignment/packing of
             * struct and union fields */
            u.s.d |= 0x80; /* "set the MSB" */
            printIntLine(u.l);
        }
    }
}

void CWE188_Reliance_On_Data_Memory_Layout__union_11_good()
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
    CWE188_Reliance_On_Data_Memory_Layout__union_11_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE188_Reliance_On_Data_Memory_Layout__union_11_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
