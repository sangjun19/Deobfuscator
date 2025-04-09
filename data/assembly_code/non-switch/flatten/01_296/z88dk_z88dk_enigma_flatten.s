	.file	"z88dk_z88dk_enigma_flatten.c"
	.text
	.globl	rings
	.bss
	.type	rings, @object
	.size	rings, 3
rings:
	.zero	3
	.globl	_TIG_IZ_3zRh_argv
	.align 8
	.type	_TIG_IZ_3zRh_argv, @object
	.size	_TIG_IZ_3zRh_argv, 8
_TIG_IZ_3zRh_argv:
	.zero	8
	.globl	notch
	.type	notch, @object
	.size	notch, 6
notch:
	.zero	6
	.globl	order
	.type	order, @object
	.size	order, 3
order:
	.zero	3
	.globl	flag
	.align 4
	.type	flag, @object
	.size	flag, 4
flag:
	.zero	4
	.globl	rotor
	.align 32
	.type	rotor, @object
	.size	rotor, 131
rotor:
	.zero	131
	.globl	pos
	.type	pos, @object
	.size	pos, 3
pos:
	.zero	3
	.globl	_TIG_IZ_3zRh_envp
	.align 8
	.type	_TIG_IZ_3zRh_envp, @object
	.size	_TIG_IZ_3zRh_envp, 8
_TIG_IZ_3zRh_envp:
	.zero	8
	.globl	plug
	.type	plug, @object
	.size	plug, 5
plug:
	.zero	5
	.globl	_TIG_IZ_3zRh_argc
	.align 4
	.type	_TIG_IZ_3zRh_argc, @object
	.size	_TIG_IZ_3zRh_argc, 4
_TIG_IZ_3zRh_argc:
	.zero	4
	.globl	ref
	.align 16
	.type	ref, @object
	.size	ref, 27
ref:
	.zero	27
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter text to be (de)coded, finish with a ."
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movb	$65, plug(%rip)
	movb	$77, 1+plug(%rip)
	movb	$84, 2+plug(%rip)
	movb	$69, 3+plug(%rip)
	movb	$0, 4+plug(%rip)
	nop
.L2:
	movl	$0, -36(%rbp)
	jmp	.L3
.L4:
	movl	-36(%rbp), %eax
	cltq
	leaq	pos(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -36(%rbp)
.L3:
	cmpl	$2, -36(%rbp)
	jle	.L4
	nop
.L5:
	movb	$87, rings(%rip)
	movb	$88, 1+rings(%rip)
	movb	$84, 2+rings(%rip)
	nop
.L6:
	movb	$3, order(%rip)
	movb	$1, 1+order(%rip)
	movb	$2, 2+order(%rip)
	nop
.L7:
	movl	$0, flag(%rip)
	nop
.L8:
	movb	$81, notch(%rip)
	movb	$69, 1+notch(%rip)
	movb	$86, 2+notch(%rip)
	movb	$74, 3+notch(%rip)
	movb	$90, 4+notch(%rip)
	movb	$0, 5+notch(%rip)
	nop
.L9:
	movb	$89, ref(%rip)
	movb	$82, 1+ref(%rip)
	movb	$85, 2+ref(%rip)
	movb	$72, 3+ref(%rip)
	movb	$81, 4+ref(%rip)
	movb	$83, 5+ref(%rip)
	movb	$76, 6+ref(%rip)
	movb	$68, 7+ref(%rip)
	movb	$80, 8+ref(%rip)
	movb	$88, 9+ref(%rip)
	movb	$78, 10+ref(%rip)
	movb	$71, 11+ref(%rip)
	movb	$79, 12+ref(%rip)
	movb	$75, 13+ref(%rip)
	movb	$77, 14+ref(%rip)
	movb	$73, 15+ref(%rip)
	movb	$69, 16+ref(%rip)
	movb	$66, 17+ref(%rip)
	movb	$70, 18+ref(%rip)
	movb	$90, 19+ref(%rip)
	movb	$67, 20+ref(%rip)
	movb	$87, 21+ref(%rip)
	movb	$86, 22+ref(%rip)
	movb	$74, 23+ref(%rip)
	movb	$65, 24+ref(%rip)
	movb	$84, 25+ref(%rip)
	movb	$0, 26+ref(%rip)
	nop
.L10:
	movb	$69, rotor(%rip)
	movb	$75, 1+rotor(%rip)
	movb	$77, 2+rotor(%rip)
	movb	$70, 3+rotor(%rip)
	movb	$76, 4+rotor(%rip)
	movb	$71, 5+rotor(%rip)
	movb	$68, 6+rotor(%rip)
	movb	$81, 7+rotor(%rip)
	movb	$86, 8+rotor(%rip)
	movb	$90, 9+rotor(%rip)
	movb	$78, 10+rotor(%rip)
	movb	$84, 11+rotor(%rip)
	movb	$79, 12+rotor(%rip)
	movb	$87, 13+rotor(%rip)
	movb	$89, 14+rotor(%rip)
	movb	$72, 15+rotor(%rip)
	movb	$88, 16+rotor(%rip)
	movb	$85, 17+rotor(%rip)
	movb	$83, 18+rotor(%rip)
	movb	$80, 19+rotor(%rip)
	movb	$65, 20+rotor(%rip)
	movb	$73, 21+rotor(%rip)
	movb	$66, 22+rotor(%rip)
	movb	$82, 23+rotor(%rip)
	movb	$67, 24+rotor(%rip)
	movb	$74, 25+rotor(%rip)
	movb	$65, 26+rotor(%rip)
	movb	$74, 27+rotor(%rip)
	movb	$68, 28+rotor(%rip)
	movb	$75, 29+rotor(%rip)
	movb	$83, 30+rotor(%rip)
	movb	$73, 31+rotor(%rip)
	movb	$82, 32+rotor(%rip)
	movb	$85, 33+rotor(%rip)
	movb	$88, 34+rotor(%rip)
	movb	$66, 35+rotor(%rip)
	movb	$76, 36+rotor(%rip)
	movb	$72, 37+rotor(%rip)
	movb	$87, 38+rotor(%rip)
	movb	$84, 39+rotor(%rip)
	movb	$77, 40+rotor(%rip)
	movb	$67, 41+rotor(%rip)
	movb	$81, 42+rotor(%rip)
	movb	$71, 43+rotor(%rip)
	movb	$90, 44+rotor(%rip)
	movb	$78, 45+rotor(%rip)
	movb	$80, 46+rotor(%rip)
	movb	$89, 47+rotor(%rip)
	movb	$70, 48+rotor(%rip)
	movb	$86, 49+rotor(%rip)
	movb	$79, 50+rotor(%rip)
	movb	$69, 51+rotor(%rip)
	movb	$66, 52+rotor(%rip)
	movb	$68, 53+rotor(%rip)
	movb	$70, 54+rotor(%rip)
	movb	$72, 55+rotor(%rip)
	movb	$74, 56+rotor(%rip)
	movb	$76, 57+rotor(%rip)
	movb	$67, 58+rotor(%rip)
	movb	$80, 59+rotor(%rip)
	movb	$82, 60+rotor(%rip)
	movb	$84, 61+rotor(%rip)
	movb	$88, 62+rotor(%rip)
	movb	$86, 63+rotor(%rip)
	movb	$90, 64+rotor(%rip)
	movb	$78, 65+rotor(%rip)
	movb	$89, 66+rotor(%rip)
	movb	$69, 67+rotor(%rip)
	movb	$73, 68+rotor(%rip)
	movb	$87, 69+rotor(%rip)
	movb	$71, 70+rotor(%rip)
	movb	$65, 71+rotor(%rip)
	movb	$75, 72+rotor(%rip)
	movb	$77, 73+rotor(%rip)
	movb	$85, 74+rotor(%rip)
	movb	$83, 75+rotor(%rip)
	movb	$81, 76+rotor(%rip)
	movb	$79, 77+rotor(%rip)
	movb	$69, 78+rotor(%rip)
	movb	$83, 79+rotor(%rip)
	movb	$79, 80+rotor(%rip)
	movb	$86, 81+rotor(%rip)
	movb	$80, 82+rotor(%rip)
	movb	$90, 83+rotor(%rip)
	movb	$74, 84+rotor(%rip)
	movb	$65, 85+rotor(%rip)
	movb	$89, 86+rotor(%rip)
	movb	$81, 87+rotor(%rip)
	movb	$85, 88+rotor(%rip)
	movb	$73, 89+rotor(%rip)
	movb	$82, 90+rotor(%rip)
	movb	$72, 91+rotor(%rip)
	movb	$88, 92+rotor(%rip)
	movb	$76, 93+rotor(%rip)
	movb	$78, 94+rotor(%rip)
	movb	$70, 95+rotor(%rip)
	movb	$84, 96+rotor(%rip)
	movb	$71, 97+rotor(%rip)
	movb	$75, 98+rotor(%rip)
	movb	$68, 99+rotor(%rip)
	movb	$67, 100+rotor(%rip)
	movb	$77, 101+rotor(%rip)
	movb	$87, 102+rotor(%rip)
	movb	$66, 103+rotor(%rip)
	movb	$86, 104+rotor(%rip)
	movb	$90, 105+rotor(%rip)
	movb	$66, 106+rotor(%rip)
	movb	$82, 107+rotor(%rip)
	movb	$71, 108+rotor(%rip)
	movb	$73, 109+rotor(%rip)
	movb	$84, 110+rotor(%rip)
	movb	$89, 111+rotor(%rip)
	movb	$85, 112+rotor(%rip)
	movb	$80, 113+rotor(%rip)
	movb	$83, 114+rotor(%rip)
	movb	$68, 115+rotor(%rip)
	movb	$78, 116+rotor(%rip)
	movb	$72, 117+rotor(%rip)
	movb	$76, 118+rotor(%rip)
	movb	$88, 119+rotor(%rip)
	movb	$65, 120+rotor(%rip)
	movb	$87, 121+rotor(%rip)
	movb	$77, 122+rotor(%rip)
	movb	$74, 123+rotor(%rip)
	movb	$81, 124+rotor(%rip)
	movb	$79, 125+rotor(%rip)
	movb	$70, 126+rotor(%rip)
	movb	$69, 127+rotor(%rip)
	movb	$67, 128+rotor(%rip)
	movb	$75, 129+rotor(%rip)
	movb	$0, 130+rotor(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_3zRh_envp(%rip)
	nop
.L12:
	movq	$0, _TIG_IZ_3zRh_argv(%rip)
	nop
.L13:
	movl	$0, _TIG_IZ_3zRh_argc(%rip)
	nop
	nop
.L14:
.L15:
#APP
# 355 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3zRh--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_3zRh_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_3zRh_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_3zRh_envp(%rip)
	nop
	movq	$14, -8(%rbp)
.L155:
	cmpq	$96, -8(%rbp)
	ja	.L156
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L18(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L18(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L18:
	.long	.L95-.L18
	.long	.L94-.L18
	.long	.L93-.L18
	.long	.L92-.L18
	.long	.L91-.L18
	.long	.L90-.L18
	.long	.L89-.L18
	.long	.L88-.L18
	.long	.L87-.L18
	.long	.L86-.L18
	.long	.L156-.L18
	.long	.L85-.L18
	.long	.L84-.L18
	.long	.L83-.L18
	.long	.L82-.L18
	.long	.L81-.L18
	.long	.L156-.L18
	.long	.L80-.L18
	.long	.L79-.L18
	.long	.L78-.L18
	.long	.L77-.L18
	.long	.L76-.L18
	.long	.L75-.L18
	.long	.L74-.L18
	.long	.L156-.L18
	.long	.L73-.L18
	.long	.L72-.L18
	.long	.L71-.L18
	.long	.L156-.L18
	.long	.L70-.L18
	.long	.L69-.L18
	.long	.L68-.L18
	.long	.L67-.L18
	.long	.L66-.L18
	.long	.L156-.L18
	.long	.L156-.L18
	.long	.L65-.L18
	.long	.L64-.L18
	.long	.L63-.L18
	.long	.L62-.L18
	.long	.L61-.L18
	.long	.L60-.L18
	.long	.L59-.L18
	.long	.L156-.L18
	.long	.L156-.L18
	.long	.L58-.L18
	.long	.L57-.L18
	.long	.L56-.L18
	.long	.L55-.L18
	.long	.L54-.L18
	.long	.L53-.L18
	.long	.L156-.L18
	.long	.L52-.L18
	.long	.L51-.L18
	.long	.L156-.L18
	.long	.L50-.L18
	.long	.L49-.L18
	.long	.L48-.L18
	.long	.L156-.L18
	.long	.L47-.L18
	.long	.L46-.L18
	.long	.L156-.L18
	.long	.L45-.L18
	.long	.L44-.L18
	.long	.L43-.L18
	.long	.L42-.L18
	.long	.L41-.L18
	.long	.L40-.L18
	.long	.L39-.L18
	.long	.L38-.L18
	.long	.L156-.L18
	.long	.L37-.L18
	.long	.L36-.L18
	.long	.L35-.L18
	.long	.L156-.L18
	.long	.L34-.L18
	.long	.L33-.L18
	.long	.L32-.L18
	.long	.L31-.L18
	.long	.L30-.L18
	.long	.L29-.L18
	.long	.L28-.L18
	.long	.L156-.L18
	.long	.L27-.L18
	.long	.L26-.L18
	.long	.L25-.L18
	.long	.L24-.L18
	.long	.L156-.L18
	.long	.L23-.L18
	.long	.L156-.L18
	.long	.L22-.L18
	.long	.L21-.L18
	.long	.L156-.L18
	.long	.L156-.L18
	.long	.L20-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L79:
	movl	$0, -28(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L96
.L53:
	movzbl	1+pos(%rip), %edx
	movzbl	1+order(%rip), %eax
	movzbl	%al, %eax
	subl	$1, %eax
	cltq
	leaq	notch(%rip), %rcx
	movzbl	(%rax,%rcx), %eax
	cmpb	%al, %dl
	jne	.L97
	movq	$33, -8(%rbp)
	jmp	.L96
.L97:
	movq	$38, -8(%rbp)
	jmp	.L96
.L29:
	movq	$65, -8(%rbp)
	jmp	.L96
.L73:
	movl	-32(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edx
	leaq	rings(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	subl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L96
.L54:
	movl	-32(%rbp), %eax
	leaq	rings(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	subl	%eax, -20(%rbp)
	movq	$56, -8(%rbp)
	jmp	.L96
.L52:
	movl	-32(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edx
	leaq	order(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	subl	$1, %eax
	imull	$26, %eax, %eax
	movl	%eax, %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %edx
	leaq	rotor(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	cmpl	%eax, -20(%rbp)
	jne	.L99
	movq	$26, -8(%rbp)
	jmp	.L96
.L99:
	movq	$15, -8(%rbp)
	jmp	.L96
.L91:
	addl	$2, -32(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L96
.L69:
	movl	$0, -32(%rbp)
	movq	$57, -8(%rbp)
	jmp	.L96
.L45:
	addl	$26, -20(%rbp)
	movq	$48, -8(%rbp)
	jmp	.L96
.L82:
	movq	$41, -8(%rbp)
	jmp	.L96
.L81:
	addl	$1, -28(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L96
.L49:
	cmpl	$64, -20(%rbp)
	jg	.L101
	movq	$96, -8(%rbp)
	jmp	.L96
.L101:
	movq	$71, -8(%rbp)
	jmp	.L96
.L30:
	movzbl	pos(%rip), %eax
	subl	$26, %eax
	movb	%al, pos(%rip)
	movq	$85, -8(%rbp)
	jmp	.L96
.L68:
	addl	$1, -24(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$90, -8(%rbp)
	jmp	.L96
.L84:
	cmpl	$46, -20(%rbp)
	jne	.L103
	movq	$83, -8(%rbp)
	jmp	.L96
.L103:
	movq	$11, -8(%rbp)
	jmp	.L96
.L38:
	movzbl	pos(%rip), %eax
	cmpb	$90, %al
	jbe	.L105
	movq	$79, -8(%rbp)
	jmp	.L96
.L105:
	movq	$85, -8(%rbp)
	jmp	.L96
.L87:
	cmpl	$64, -20(%rbp)
	jg	.L107
	movq	$60, -8(%rbp)
	jmp	.L96
.L107:
	movq	$18, -8(%rbp)
	jmp	.L96
.L17:
	addl	$26, -20(%rbp)
	movq	$71, -8(%rbp)
	jmp	.L96
.L58:
	movl	-32(%rbp), %eax
	leaq	plug(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	movl	%eax, -20(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L96
.L31:
	movl	-20(%rbp), %eax
	subl	$65, %eax
	cltq
	leaq	ref(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	movl	%eax, -20(%rbp)
	movl	$3, -32(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L96
.L94:
	movl	-32(%rbp), %eax
	leaq	plug(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	testb	%al, %al
	je	.L109
	movq	$7, -8(%rbp)
	jmp	.L96
.L109:
	movq	$31, -8(%rbp)
	jmp	.L96
.L28:
	movzbl	1+pos(%rip), %eax
	addl	$1, %eax
	movb	%al, 1+pos(%rip)
	movq	$73, -8(%rbp)
	jmp	.L96
.L74:
	movzbl	1+pos(%rip), %eax
	cmpb	$90, %al
	jbe	.L111
	movq	$72, -8(%rbp)
	jmp	.L96
.L111:
	movq	$88, -8(%rbp)
	jmp	.L96
.L32:
	cmpl	$90, -20(%rbp)
	jle	.L113
	movq	$39, -8(%rbp)
	jmp	.L96
.L113:
	movq	$49, -8(%rbp)
	jmp	.L96
.L92:
	movl	-32(%rbp), %eax
	leaq	plug(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	cmpl	%eax, -20(%rbp)
	jne	.L115
	movq	$5, -8(%rbp)
	jmp	.L96
.L115:
	movq	$21, -8(%rbp)
	jmp	.L96
.L76:
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edx
	leaq	plug(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	cmpl	%eax, -20(%rbp)
	jne	.L117
	movq	$45, -8(%rbp)
	jmp	.L96
.L117:
	movq	$20, -8(%rbp)
	jmp	.L96
.L20:
	movl	-32(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edx
	leaq	pos(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	addl	%eax, -20(%rbp)
	movq	$67, -8(%rbp)
	jmp	.L96
.L65:
	cmpl	$90, -20(%rbp)
	jle	.L119
	movq	$53, -8(%rbp)
	jmp	.L96
.L119:
	movq	$17, -8(%rbp)
	jmp	.L96
.L33:
	movzbl	pos(%rip), %eax
	addl	$1, %eax
	movb	%al, pos(%rip)
	movq	$69, -8(%rbp)
	jmp	.L96
.L48:
	cmpl	$2, -32(%rbp)
	ja	.L121
	movq	$6, -8(%rbp)
	jmp	.L96
.L121:
	movq	$78, -8(%rbp)
	jmp	.L96
.L39:
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	imulq	$702812831, %rax, %rax
	shrq	$32, %rax
	movq	%rax, %rcx
	movl	%edx, %eax
	subl	%ecx, %eax
	shrl	%eax
	addl	%ecx, %eax
	shrl	$5, %eax
	imull	$55, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L123
	movq	$91, -8(%rbp)
	jmp	.L96
.L123:
	movq	$66, -8(%rbp)
	jmp	.L96
.L25:
	movl	flag(%rip), %eax
	testl	%eax, %eax
	je	.L125
	movq	$9, -8(%rbp)
	jmp	.L96
.L125:
	movq	$13, -8(%rbp)
	jmp	.L96
.L72:
	movl	-28(%rbp), %eax
	addl	$65, %eax
	movl	%eax, -20(%rbp)
	movl	-32(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edx
	leaq	rings(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	addl	%eax, -20(%rbp)
	movq	$36, -8(%rbp)
	jmp	.L96
.L85:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$64, -8(%rbp)
	jmp	.L96
.L86:
	movzbl	1+pos(%rip), %eax
	addl	$1, %eax
	movb	%al, 1+pos(%rip)
	movq	$23, -8(%rbp)
	jmp	.L96
.L83:
	movzbl	pos(%rip), %edx
	movzbl	order(%rip), %eax
	movzbl	%al, %eax
	subl	$1, %eax
	cltq
	leaq	notch(%rip), %rcx
	movzbl	(%rax,%rcx), %eax
	cmpb	%al, %dl
	jne	.L127
	movq	$81, -8(%rbp)
	jmp	.L96
.L127:
	movq	$38, -8(%rbp)
	jmp	.L96
.L44:
	subl	$26, -20(%rbp)
	movq	$42, -8(%rbp)
	jmp	.L96
.L78:
	movzbl	2+pos(%rip), %eax
	cmpb	$90, %al
	jbe	.L129
	movq	$59, -8(%rbp)
	jmp	.L96
.L129:
	movq	$86, -8(%rbp)
	jmp	.L96
.L67:
	movl	-32(%rbp), %eax
	leaq	plug(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	testb	%al, %al
	je	.L131
	movq	$3, -8(%rbp)
	jmp	.L96
.L131:
	movq	$30, -8(%rbp)
	jmp	.L96
.L80:
	movl	-32(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edx
	leaq	pos(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	subl	%eax, -20(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L96
.L22:
	movl	-24(%rbp), %ecx
	movl	%ecx, %edx
	movl	$3435973837, %eax
	imulq	%rdx, %rax
	shrq	$32, %rax
	shrl	$2, %eax
	movl	%eax, %edx
	sall	$2, %edx
	addl	%eax, %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	testl	%eax, %eax
	jne	.L133
	movq	$68, -8(%rbp)
	jmp	.L96
.L133:
	movq	$80, -8(%rbp)
	jmp	.L96
.L61:
	cmpl	$64, -20(%rbp)
	jg	.L135
	movq	$62, -8(%rbp)
	jmp	.L96
.L135:
	movq	$48, -8(%rbp)
	jmp	.L96
.L40:
	cmpl	$90, -20(%rbp)
	jle	.L137
	movq	$47, -8(%rbp)
	jmp	.L96
.L137:
	movq	$25, -8(%rbp)
	jmp	.L96
.L50:
	cmpl	$64, -20(%rbp)
	jg	.L139
	movq	$46, -8(%rbp)
	jmp	.L96
.L139:
	movq	$29, -8(%rbp)
	jmp	.L96
.L46:
	addl	$26, -20(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L96
.L47:
	movzbl	2+pos(%rip), %eax
	subl	$26, %eax
	movb	%al, 2+pos(%rip)
	movq	$86, -8(%rbp)
	jmp	.L96
.L89:
	movl	-32(%rbp), %eax
	leaq	pos(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	addl	%eax, -20(%rbp)
	movq	$77, -8(%rbp)
	jmp	.L96
.L71:
	cmpl	$25, -28(%rbp)
	ja	.L141
	movq	$52, -8(%rbp)
	jmp	.L96
.L141:
	movq	$26, -8(%rbp)
	jmp	.L96
.L63:
	movl	$0, -32(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L96
.L26:
	cmpl	$0, -32(%rbp)
	je	.L143
	movq	$94, -8(%rbp)
	jmp	.L96
.L143:
	movq	$95, -8(%rbp)
	jmp	.L96
.L34:
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edx
	leaq	plug(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	movl	%eax, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L96
.L55:
	subl	$1, -32(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L96
.L37:
	movl	-32(%rbp), %eax
	leaq	order(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	subl	$1, %eax
	imull	$26, %eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	subl	$65, %eax
	cltq
	leaq	rotor(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	movl	%eax, -20(%rbp)
	movl	-32(%rbp), %eax
	leaq	rings(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	addl	%eax, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L96
.L75:
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edx
	leaq	plug(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	cmpl	%eax, -20(%rbp)
	jne	.L145
	movq	$37, -8(%rbp)
	jmp	.L96
.L145:
	movq	$4, -8(%rbp)
	jmp	.L96
.L51:
	subl	$26, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L96
.L42:
	call	getchar@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -20(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L96
.L56:
	subl	$26, -20(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L96
.L35:
	movzbl	1+pos(%rip), %eax
	cmpb	$90, %al
	jbe	.L147
	movq	$0, -8(%rbp)
	jmp	.L96
.L147:
	movq	$50, -8(%rbp)
	jmp	.L96
.L90:
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edx
	leaq	plug(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	movl	%eax, -20(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L96
.L21:
	movl	$10, %edi
	call	putchar@PLT
	movq	$80, -8(%rbp)
	jmp	.L96
.L36:
	movzbl	1+pos(%rip), %eax
	subl	$26, %eax
	movb	%al, 1+pos(%rip)
	movq	$88, -8(%rbp)
	jmp	.L96
.L66:
	movl	$1, flag(%rip)
	movq	$38, -8(%rbp)
	jmp	.L96
.L64:
	movl	-32(%rbp), %eax
	leaq	plug(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	movl	%eax, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L96
.L43:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movl	-20(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	jne	.L149
	movq	$80, -8(%rbp)
	jmp	.L96
.L149:
	movq	$76, -8(%rbp)
	jmp	.L96
.L60:
	movl	$0, -24(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movb	$65, pos(%rip)
	movb	$87, 1+pos(%rip)
	movb	$69, 2+pos(%rip)
	movq	$65, -8(%rbp)
	jmp	.L96
.L19:
	movl	$0, -32(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L96
.L59:
	movl	-32(%rbp), %eax
	leaq	pos(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	subl	$65, %eax
	subl	%eax, -20(%rbp)
	movq	$55, -8(%rbp)
	jmp	.L96
.L95:
	movzbl	1+pos(%rip), %eax
	subl	$26, %eax
	movb	%al, 1+pos(%rip)
	movq	$50, -8(%rbp)
	jmp	.L96
.L57:
	addl	$26, -20(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L96
.L62:
	subl	$26, -20(%rbp)
	movq	$49, -8(%rbp)
	jmp	.L96
.L41:
	movl	$32, %edi
	call	putchar@PLT
	movq	$80, -8(%rbp)
	jmp	.L96
.L27:
	movl	$0, %edi
	call	exit@PLT
.L88:
	movl	-32(%rbp), %eax
	leaq	plug(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %eax
	cmpl	%eax, -20(%rbp)
	jne	.L151
	movq	$75, -8(%rbp)
	jmp	.L96
.L151:
	movq	$22, -8(%rbp)
	jmp	.L96
.L23:
	movzbl	2+pos(%rip), %eax
	addl	$1, %eax
	movb	%al, 2+pos(%rip)
	movq	$19, -8(%rbp)
	jmp	.L96
.L70:
	addl	$1, -32(%rbp)
	movq	$57, -8(%rbp)
	jmp	.L96
.L24:
	movl	$0, flag(%rip)
	movq	$13, -8(%rbp)
	jmp	.L96
.L93:
	cmpl	$90, -20(%rbp)
	jle	.L153
	movq	$63, -8(%rbp)
	jmp	.L96
.L153:
	movq	$42, -8(%rbp)
	jmp	.L96
.L77:
	addl	$2, -32(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L96
.L156:
	nop
.L96:
	jmp	.L155
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
