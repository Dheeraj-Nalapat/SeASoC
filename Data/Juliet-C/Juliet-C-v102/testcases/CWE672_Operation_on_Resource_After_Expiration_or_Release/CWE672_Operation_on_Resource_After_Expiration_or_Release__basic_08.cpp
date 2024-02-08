/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE672_Operation_on_Resource_After_Expiration_or_Release__basic_08.cpp
Label Definition File: CWE672_Operation_on_Resource_After_Expiration_or_Release__basic.label.xml
Template File: sources-sinks-08.tmpl.cpp
*/
/*
 * @description
 * CWE: 672 Operation on Resource After Expiration or Release
 * BadSource:  Add values to the list, including the number zero
 * GoodSource: Add value to the list that are not zero
 * Sinks:
 *    GoodSink: Iterate through the list without attempting to clear its contents
 *    BadSink : Iterate through the list, but clear the list if it contains a zero
 * Flow Variant: 08 Control flow: if(static_returns_t()) and if(static_returns_f())
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#include <list>
#include <iostream>
using namespace std;

/* The two function below always return the same value, so a tool
   should be able to identify that calls to the functions will always
   return a fixed value. */
static int static_returns_t()
{
    return 1;
}

static int static_returns_f()
{
    return 0;
}

namespace CWE672_Operation_on_Resource_After_Expiration_or_Release__basic_08
{

#ifndef OMITBAD

void bad()
{
    list<int> data;
    if(static_returns_t())
    {
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_returns_t())
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_returns_t() to static_returns_f() */
static void goodB2G1()
{
    list<int> data;
    if(static_returns_t())
    {
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    list<int> data;
    if(static_returns_t())
    {
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_returns_t())
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_returns_t() to static_returns_f() */
static void goodG2B1()
{
    list<int> data;
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_returns_t())
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    list<int> data;
    if(static_returns_t())
    {
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    if(static_returns_t())
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

void good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE672_Operation_on_Resource_After_Expiration_or_Release__basic_08; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
