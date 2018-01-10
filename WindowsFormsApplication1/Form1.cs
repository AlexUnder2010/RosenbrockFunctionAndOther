using System;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            //функцiя Розенброка 
            FunctionWrapper.Function rozenblocksFunction =
                x => 100 * Math.Pow(x[1] - Math.Pow(x[0], 2), 2) + Math.Pow(1 - x[0], 2);

            var rozenblocksCounter = new FunctionCallsCounter(rozenblocksFunction);

            var rozenblocksFunctionWrapper = new FunctionWrapper(
                function: rozenblocksCounter.GetFunction(),
                gradient: x => new Vector(
                    //похідна по х1 функції Розенброка
                    400 * Math.Pow(x[0], 3) - 400 * x[0] * x[1] + 2 * x[0] - 2,
                    //похідна по х2 функції Розенброка
                    200 * x[1] - 200 * Math.Pow(x[0], 2)
                    )
                );

            //початкове наближення для функцiї Розенброка
            var x0ForRozenboksFunction = new Point(0, 0);


            //функцiя iндивiдуального завдання для пошуку локального мiнiмуму
            //за методом найшвидшого спуску та методом покоординатного спуску Гауса-Зейделя
            FunctionWrapper.Function firstFunction =
                x => x[0] * x[0] - 2 * x[0] * x[1] + 3 * x[1] * x[1] + x[0] - 4 * x[1];

            var firstCounter = new FunctionCallsCounter(firstFunction);

            var firstFunctionWrapper = new FunctionWrapper(
                function: firstCounter.GetFunction(),
                gradient: x => new Vector(
                    //похідна по х1 функції найшвидшого спуску
                    2 * x[0] - 2 * x[1] + 1,
                    //похідна по х1 функції найшвидшого спуску
                    -2 * (x[0] - 3 * x[1] + 2)
                    )
                );

            //початкове наближення для першої функцiї iндивiдуального завдання
            var x0ForFirstFunction = new Point(1, 1);

            //функцiя iндивiдуального завдання третього варiанту для пошуку локального мiнiмуму
            //за методом Ньютона та модифiкованим узагальненим методом Ньютона
            FunctionWrapper.Function secondFunction =
                x => 3 * Math.Pow(x[0], 2) + Math.Pow(x[1], 2) + Math.Exp(x[0] * x[0] + x[1] * x[1]) - 2 * x[0] + x[1];

            var secondCounter = new FunctionCallsCounter(secondFunction);

            var secondFunctionWrapper = new FunctionWrapper(
                function: secondCounter.GetFunction(),
                gradient: x => new Vector(
                    //похідна по х1 функції до 2 завдання
                    2 * x[0] * (Math.Exp(x[0] * x[0] + x[1] * x[1]) + 3) - 2,
                    //похідна по х2 функції до 2 завдання
                    2 * x[1] * (Math.Exp(x[0] * x[0] + x[1] * x[1]) + 1) + 1
                    ),
                //матриця Гессе
                // func 3*Pow(x1, 2) + Pow(x2, 2) + Exp(x1 * x1 + x2 * x2) - 2*x1 + x2
                hessian: x => new SquareMatrix(new double[2][]{
                        new double[2]{
                            2*((2*x[0]*x[0]+1)*Math.Exp(x[0]*x[0] + x[1]*x[1])+3),
                            4*x[0]*x[1]*Math.Exp(x[0]*x[0] + x[1]*x[1])
                        },
                        new double[2]{
                            4*x[0]*x[1]*Math.Exp(x[0]*x[0] + x[1]*x[1]),
                            2*((2*x[1]*x[1]+1)*Math.Exp(x[0]*x[0] + x[1]*x[1])+1)}
                    })
                );

            //початкове наближення для другої функцiї iндивiдуального завдання
            var x0ForSecondFunction = new Point(1, 1);

            const double h0 = 1;
        }

        public class Point
        {
            private readonly double[] _values;

            public Point(params double[] values)
            {
                _values = values;
            }

            public int Length
            {
                get { return _values.Length; }
            }

            //доступ до координати за iндексом
            public double this[int i]
            {
                get
                {
                    if (i >= _values.Length)
                        throw new Exception("Неправильний iндекс");
                    return _values[i];
                }
                set { _values[i] = value; }
            }

            //рiзниця точок - вектор
            public static Vector operator -(Point leftPoint, Point rightPoint)
            {
                if (leftPoint.Length != rightPoint.Length)
                    throw new Exception("Рiзна розмiрнiсть точок");

                var resultVector = new double[leftPoint.Length];
                for (var i = 0; i < leftPoint.Length; i++)
                {
                    resultVector[i] = leftPoint[i] - rightPoint[i];
                }

                return new Vector(resultVector);

            }

            //рiзниця точки i вектора - нова точка
            public static Point operator -(Point point, Vector vector)
            {
                if (point.Length != vector.Length)
                    throw new Exception("Рiзна розмiрнiсть точки i вектора");

                var resultPoint = new double[point.Length];
                for (var i = 0; i < point.Length; i++)
                {
                    resultPoint[i] = point[i] - vector[i];
                }

                return new Point(resultPoint);

            }

            public Point Clone()
            {
                return new Point((double[]) _values.Clone());
            }

            public override string ToString()
            {
                var res = new StringBuilder();
                res.Append("( ");
                for (var i = 0; i < _values.Length - 1; i++)
                {
                    res.Append(_values[i] + ", ");
                }
                res.Append(_values[_values.Length - 1]);
                res.Append(" )");
                return res.ToString();
            }
        }

        public class Vector
        {
            private readonly double[] _values;

            public Vector(params double[] values)
            {
                _values = values;
            }

            //нормалiзацiя
            public double Normalize()
            {
                return Math.Sqrt(_values.Sum(t => Math.Pow(t, 2)));
            }

            //розмiрнiсть вектора
            public int Length
            {
                get { return _values.Length; }
            }

            //доступ до координати з iндексом
            public double this[int i]
            {
                get
                {
                    if (i >= _values.Length)
                        throw new Exception("Неправильний iндекс");
                    return _values[i];
                }
                set { _values[i] = value; }
            }

            //множення числа на вектор
            public static Vector operator *(double c, Vector v)
            {
                var resultVector = new double[v.Length];
                for (var i = 0; i < v.Length; i++)
                {
                    resultVector[i] = c*v[i];
                }
                return new Vector(resultVector);
            }

            //множення вектора на число
            public static Vector operator *(Vector v, double c)
            {
                return c*v;
            }

        }

        public class SquareMatrix
        {
            private readonly double[][] _values;

            public SquareMatrix(double[][] value)
            {
                _values = value;
            }

            public SquareMatrix InvertByJordanGauss()
            {

                var input = new double[_values.Length][];
                for (var i = 0; i < _values.Length; i++)
                {
                    input[i] = (double[]) _values[i].Clone();
                }

                var result = new double[input.Length][];
                for (var i = 0; i < result.Length; i++)
                {
                    result[i] = new double[input.Length];
                    for (var j = 0; j < result[0].Length; j++)
                    {
                        result[i][j] = (i == j) ? 1 : 0;
                    }
                }

                double div;

                for (var i = 0; i < result[0].Length - 1; i++)
                {

                    div = input[i][i];
                    //утворення 1 на дiагоналi
                    for (var j = 0; j < result[0].Length; j++)
                    {
                        result[i][j] /= div;
                        input[i][j] /= div;
                    }

                    //утворення 0 у стовпцях
                    for (var j = i + 1; j < result.Length; j++)
                    {

                        div = input[j][i];
                        for (var k = 0; k < result[0].Length; k++)
                        {
                            result[j][k] -= result[i][k]*div;
                            input[j][k] -= input[i][k]*div;
                        }
                    }
                }

                div = input[input.Length - 1][input[0].Length - 1];
                //утворення 1 на останнiй дiагоналi
                for (var j = 0; j < result[0].Length; j++)
                {
                    result[input.Length - 1][j] /= div;
                    input[input.Length - 1][j] /= div;
                }

                //Обернений хiд алгоритму(змiнюємо числа над дiагоналлю)
                for (var i = result.Length - 1; i > 0; i--)
                {

                    for (var j = i - 1; j >= 0; j--)
                    {

                        div = input[j][i];
                        for (var k = 0; k < result[0].Length; k++)
                        {
                            result[j][k] -= result[i][k]*div;
                            input[j][k] -= input[i][k]*div;
                        }
                    }
                }

                return new SquareMatrix(result);
            }

            public double this[int i, int j]
            {
                get
                {
                    if (i >= _values.Length || j >= _values.Length)
                        throw new Exception("Неправильний iндекс");
                    return _values[i][j];
                }
                set { _values[i][j] = value; }
            }

        }

        public class FunctionWrapper
        {

            public delegate double Function(Point p);

            public delegate Vector Gradient(Point p);

            public delegate SquareMatrix Hessian(Point p);

            //збереження функцiї
            private readonly Function _function;

            //збереження градiєнта функцiї
            private readonly Gradient _gradient;

            //збереження гессiана функцiї
            private readonly Hessian _hessian;

            public FunctionWrapper(Function function, Gradient gradient, Hessian hessian = null)
            {
                _function = function;
                _gradient = gradient;
                _hessian = hessian;
            }

            //знаходження значення функцiї в точцi
            public double Eval(Point p)
            {
                return _function(p);
            }

            //знаходження значення градiєнту функцiї в точцi
            public Vector Grad(Point p)
            {
                return _gradient(p);
            }

            //знаходження значення гессiану функцiї в точцi
            public SquareMatrix Hess(Point p)
            {
                return _hessian(p);
            }

            //реалiзацiя алгоритму обчислення крокового множника для алгоритму найшвидшого спуску
            public double FindHForFastestDescent(Point x0, Vector g, double h0, double epsilon)
            {
                var h = .0;
                var f1 = _function(x0);
                Point x2;
                double f2;
                do
                {

                    h0 /= 2;
                    x2 = x0 - h0*g;
                    f2 = _function(x2);
                } while (!(f1 > f2 || h0 < epsilon));


                if (h0 > epsilon)
                {
                    Point x1;
                    do
                    {
                        //Тiльки в режимi DEBUG
                        Program.CountThisLoop();
                        x1 = x2;
                        f1 = f2;
                        h += h0;
                        x2 = x1 - h*g;
                        f2 = _function(x2);
                    } while (!(f1 < f2));

                    var ha = h - 2*h0;
                    var hb = h;

                    var sigma = epsilon/3;

                    do
                    {
                        //Тiльки в режимi DEBUG
                        Program.CountThisLoop();
                        var h1 = (ha + hb - sigma)/2;
                        var h2 = (ha + hb + sigma)/2;

                        x1 = x0 - h1*g;
                        x2 = x0 - h2*g;

                        f1 = _function(x1);
                        f2 = _function(x2);

                        if (f1 <= f2)
                        {
                            hb = h2;
                        }
                        else
                        {
                            ha = h1;
                        }

                    } while (!(hb - ha < epsilon));

                    h = (ha + hb)/2;

                }

                return h;

            }

            //реалiзацiя алгоритму обчислення крокового множника для алгоритму покоординатного спуску
            public double FindHForCoordinateDescent(Point x0, double z, int num, double h0, double epsilon)
            {
                var h = .0;
                var f1 = _function(x0);
                Point x1, x2 = x0.Clone();
                double f2;
                do
                {
                    //Тiльки в режимi DEBUG
                    Program.CountThisLoop();

                    h0 /= 2;
                    x2[num] = x0[num] + h0*z; //рiзнится з найшвидшим спуском
                    f2 = _function(x2);
                } while (!(f1 > f2 || h0 < epsilon));


                if (h0 > epsilon)
                {
                    do
                    {
                        //Тiльки в режимi DEBUG
                        Program.CountThisLoop();

                        x1 = x2.Clone();
                        f1 = f2;
                        h += h0;
                        x2[num] = x1[num] + h*z; //рiзнится з найшвидшим спуском
                        f2 = _function(x2);
                    } while (!(f1 < f2));

                    var ha = h - 2*h0;
                    var hb = h;

                    var sigma = epsilon/3;

                    do
                    {
                        //Тiльки в режимi DEBUG
                        Program.CountThisLoop();

                        var h1 = (ha + hb - sigma)/2;
                        var h2 = (ha + hb + sigma)/2;

                        x1[num] = x0[num] + h1*z; //рiзнится з найшвидшим спуском
                        x2[num] = x0[num] + h2*z; //рiзнится з найшвидшим спуском

                        f1 = _function(x1);
                        f2 = _function(x2);

                        if (f1 <= f2)
                        {
                            hb = h2;
                        }
                        else
                        {
                            ha = h1;
                        }

                    } while (!(hb - ha < epsilon));

                    h = (ha + hb)/2;

                }

                return h;

            }


        }

        public class FunctionCallsCounter
        {
            private readonly FunctionWrapper.Function _functionToCall;

            public FunctionCallsCounter(FunctionWrapper.Function functionToCall)
            {
                Count = 0;
                _functionToCall = functionToCall;
            }

            public FunctionWrapper.Function GetFunction()
            {
                return x =>
                {
                    Count++;
                    return _functionToCall(x);
                };
            }

            public int Count { get; set; }
        }

        class Program
        {

            //реалiзацiя методу найшвидшого спуску
            static Point FastestDescent(FunctionWrapper function, Point x0, double h0, double epsilon)
            {
                var g = function.Grad(x0);

                if (g.Normalize() > epsilon)
                {
                    Point x;
                    do
                    {
                        //Тiльки в режимi DEBUG
                        CountThisLoop();

                        x = x0;
                        var h = function.FindHForFastestDescent(x, g, h0, epsilon);
                        x0 = x - (h*g);
                        g = function.Grad(x0);

                    } while (!(((x - x0).Normalize() < epsilon) || (g.Normalize() < epsilon)));
                }

                return x0;

            }

            //реалiзацiя методу покоординатного спуску Гауса-Зейделя
            static Point CoordinateDescent(FunctionWrapper function, Point x0, double h0, double epsilon)
            {
                var h = new Vector(new double[x0.Length]);
                for (var i = 0; i < h.Length; i++)
                {
                    h[i] = h0;
                }

                var xInt = x0.Clone();
                Point xExt;

                do
                {
                    xExt = xInt.Clone();

                    for (var i = 0; i < xInt.Length; i++)
                    {
                        //Тiльки в режимi DEBUG
                        CountThisLoop();

                        var x = xInt.Clone();

                        var y1 = x.Clone();
                        y1[i] += 3*epsilon;
                        var y2 = x.Clone();
                        y2[i] -= 3*epsilon;
                        var f1 = function.Eval(y1);
                        var f2 = function.Eval(y2);

                        var z = Math.Sign(f2 - f1);

                        h[i] = function.FindHForCoordinateDescent(xInt.Clone(), z, i, h[i], epsilon);

                        xInt[i] = x[i] + h[i]*z;
                    }
                } while ((xInt - xExt).Normalize() >= epsilon);

                return xInt;

            }

            //реалiзацiя методу Ньютона
            static Point NewtonMethod(FunctionWrapper function, Point x0, double epsilon)
            {
                var g = function.Grad(x0);

                if (g.Normalize() > epsilon)
                {
                    do
                    {
                        //Тiльки в режимi DEBUG
                        CountThisLoop();

                        var x = x0;
                        var hessian = function.Hess(x0);
                        var hessianInverted = hessian.InvertByJordanGauss();

                        x0 = new Point(new double[x0.Length]);
                        var w = new Vector(new double[x0.Length]);

                        for (var i = 0; i < w.Length; i++)
                        {
                            for (var j = 0; j < w.Length; j++)
                            {
                                w[i] += hessianInverted[i, j]*g[j];
                            }
                        }

                        x0 = x - w;
                        g = function.Grad(x0);

                    } while (g.Normalize() >= epsilon);
                }
                return x0;

            }

            //реалiзацiя узагальненого методу Ньютона
            static Point GeneralizedNewtonMethod(FunctionWrapper function, Point x0, double h0, double epsilon)
            {
                var g = function.Grad(x0);
                double h;

                if (g.Normalize() > epsilon)
                {
                    do
                    {

                        //Тiльки в режимi DEBUG
                        CountThisLoop();

                        var x = x0;
                        var hessian = function.Hess(x0);
                        var hessianInverted = hessian.InvertByJordanGauss();

                        x0 = new Point(new double[x0.Length]);
                        var w = new Vector(new double[x0.Length]);

                        for (var i = 0; i < w.Length; i++)
                        {
                            for (var j = 0; j < w.Length; j++)
                            {
                                w[i] += hessianInverted[i, j]*g[j];
                            }
                        }

                        h = function.FindHForFastestDescent(x, w, h0, epsilon);
                        x0 = x - h*w;

                        g = function.Grad(x0);

                    } while (g.Normalize() >= epsilon && Math.Abs(h) > Math.Sqrt(epsilon));
                }
                return x0;

            }

            private static int _loopCounter;

            [Conditional("DEBUG")]
            public static void CountThisLoop()
            {
                _loopCounter++;
            }

            private void button1_Click(object sender, EventArgs e)
            {

            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Hello");
        }
    }
}
