	.file	"bharathkumar12341_FPGA_main_flatten.c"
	.text
	.globl	_TIG_IZ_i3df_argc
	.bss
	.align 4
	.type	_TIG_IZ_i3df_argc, @object
	.size	_TIG_IZ_i3df_argc, 4
_TIG_IZ_i3df_argc:
	.zero	4
	.globl	_TIG_IZ_i3df_envp
	.align 8
	.type	_TIG_IZ_i3df_envp, @object
	.size	_TIG_IZ_i3df_envp, 8
_TIG_IZ_i3df_envp:
	.zero	8
	.globl	_TIG_IZ_i3df_argv
	.align 8
	.type	_TIG_IZ_i3df_argv, @object
	.size	_TIG_IZ_i3df_argv, 8
_TIG_IZ_i3df_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"if"
.LC1:
	.string	"int"
	.text
	.globl	getNextToken
	.type	getNextToken, @function
getNextToken:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	$5, -8(%rbp)
.L76:
	cmpq	$68, -8(%rbp)
	ja	.L77
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L77-.L4
	.long	.L44-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L78-.L4
	.long	.L42-.L4
	.long	.L41-.L4
	.long	.L77-.L4
	.long	.L78-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L39-.L4
	.long	.L77-.L4
	.long	.L38-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L37-.L4
	.long	.L36-.L4
	.long	.L35-.L4
	.long	.L34-.L4
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L77-.L4
	.long	.L31-.L4
	.long	.L30-.L4
	.long	.L29-.L4
	.long	.L77-.L4
	.long	.L28-.L4
	.long	.L27-.L4
	.long	.L26-.L4
	.long	.L25-.L4
	.long	.L24-.L4
	.long	.L77-.L4
	.long	.L23-.L4
	.long	.L77-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L20-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L19-.L4
	.long	.L77-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L77-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L77-.L4
	.long	.L12-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L77-.L4
	.long	.L78-.L4
	.long	.L78-.L4
	.long	.L9-.L4
	.long	.L77-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L37:
	movq	-112(%rbp), %rax
	movl	$6, (%rax)
	movq	$61, -8(%rbp)
	jmp	.L45
.L16:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L45
.L31:
	movq	-104(%rbp), %rdx
	movl	-84(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	ungetc@PLT
	movq	-112(%rbp), %rdx
	movl	-68(%rbp), %eax
	cltq
	movb	$0, 4(%rdx,%rax)
	movq	-112(%rbp), %rax
	movl	$2, (%rax)
	movq	$60, -8(%rbp)
	jmp	.L45
.L17:
	movq	-112(%rbp), %rax
	movl	$3, (%rax)
	movq	-112(%rbp), %rax
	movb	$61, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L15:
	cmpl	$98, -80(%rbp)
	jg	.L46
	movq	$64, -8(%rbp)
	jmp	.L45
.L46:
	movq	$1, -8(%rbp)
	jmp	.L45
.L27:
	movl	-84(%rbp), %eax
	subl	$40, %eax
	cmpl	$21, %eax
	ja	.L49
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L51(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L51(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L51:
	.long	.L56-.L51
	.long	.L55-.L51
	.long	.L49-.L51
	.long	.L54-.L51
	.long	.L49-.L51
	.long	.L53-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L52-.L51
	.long	.L49-.L51
	.long	.L50-.L51
	.text
.L52:
	movq	$6, -8(%rbp)
	jmp	.L57
.L55:
	movq	$68, -8(%rbp)
	jmp	.L57
.L56:
	movq	$53, -8(%rbp)
	jmp	.L57
.L53:
	movq	$22, -8(%rbp)
	jmp	.L57
.L54:
	movq	$66, -8(%rbp)
	jmp	.L57
.L50:
	movq	$49, -8(%rbp)
	jmp	.L57
.L49:
	movq	$37, -8(%rbp)
	nop
.L57:
	jmp	.L45
.L9:
	movq	-112(%rbp), %rax
	movl	$1, (%rax)
	movq	$61, -8(%rbp)
	jmp	.L45
.L12:
	cmpl	$0, -76(%rbp)
	jne	.L58
	movq	$18, -8(%rbp)
	jmp	.L45
.L58:
	movq	$62, -8(%rbp)
	jmp	.L45
.L26:
	cmpl	$0, -72(%rbp)
	jne	.L60
	movq	$32, -8(%rbp)
	jmp	.L45
.L60:
	movq	$65, -8(%rbp)
	jmp	.L45
.L13:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -84(%rbp)
	movq	$48, -8(%rbp)
	jmp	.L45
.L44:
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -84(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L45
.L32:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L62
	movq	$11, -8(%rbp)
	jmp	.L45
.L62:
	movq	$50, -8(%rbp)
	jmp	.L45
.L34:
	movl	$0, -68(%rbp)
	movl	-68(%rbp), %eax
	movl	%eax, -60(%rbp)
	addl	$1, -68(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, %ecx
	movq	-112(%rbp), %rdx
	movl	-60(%rbp), %eax
	cltq
	movb	%cl, 4(%rdx,%rax)
	movq	$54, -8(%rbp)
	jmp	.L45
.L3:
	movq	-112(%rbp), %rax
	movl	$9, (%rax)
	movq	-112(%rbp), %rax
	movb	$41, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L30:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8, %eax
	testl	%eax, %eax
	je	.L64
	movq	$52, -8(%rbp)
	jmp	.L45
.L64:
	movq	$33, -8(%rbp)
	jmp	.L45
.L39:
	movl	$0, -80(%rbp)
	movl	-80(%rbp), %eax
	movl	%eax, -52(%rbp)
	addl	$1, -80(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, %ecx
	movq	-112(%rbp), %rdx
	movl	-52(%rbp), %eax
	cltq
	movb	%cl, 4(%rdx,%rax)
	movq	$1, -8(%rbp)
	jmp	.L45
.L38:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L66
	movq	$21, -8(%rbp)
	jmp	.L45
.L66:
	movq	$30, -8(%rbp)
	jmp	.L45
.L36:
	movl	-68(%rbp), %eax
	movl	%eax, -56(%rbp)
	addl	$1, -68(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, %ecx
	movq	-112(%rbp), %rdx
	movl	-56(%rbp), %eax
	cltq
	movb	%cl, 4(%rdx,%rax)
	movq	$54, -8(%rbp)
	jmp	.L45
.L25:
	movq	-112(%rbp), %rax
	movl	$0, (%rax)
	movq	$61, -8(%rbp)
	jmp	.L45
.L5:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -84(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L45
.L41:
	movq	-112(%rbp), %rax
	movl	$10, (%rax)
	movq	-112(%rbp), %rax
	movb	$59, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L29:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L68
	movq	$67, -8(%rbp)
	jmp	.L45
.L68:
	movq	$20, -8(%rbp)
	jmp	.L45
.L21:
	cmpl	$-1, -84(%rbp)
	je	.L70
	movq	$35, -8(%rbp)
	jmp	.L45
.L70:
	movq	$46, -8(%rbp)
	jmp	.L45
.L18:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L72
	movq	$42, -8(%rbp)
	jmp	.L45
.L72:
	movq	$25, -8(%rbp)
	jmp	.L45
.L33:
	movq	-112(%rbp), %rax
	movl	$5, (%rax)
	movq	-112(%rbp), %rax
	movb	$45, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L14:
	movq	-112(%rbp), %rax
	movl	$8, (%rax)
	movq	-112(%rbp), %rax
	movb	$40, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L7:
	movq	-112(%rbp), %rax
	addq	$4, %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -76(%rbp)
	movq	$56, -8(%rbp)
	jmp	.L45
.L42:
	movq	$67, -8(%rbp)
	jmp	.L45
.L24:
	movq	-104(%rbp), %rdx
	movl	-84(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	ungetc@PLT
	movq	-112(%rbp), %rdx
	movl	-80(%rbp), %eax
	cltq
	movb	$0, 4(%rdx,%rax)
	movq	-112(%rbp), %rax
	addq	$4, %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -72(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L45
.L22:
	movq	-112(%rbp), %rax
	movl	$12, (%rax)
	movl	-84(%rbp), %eax
	movl	%eax, %edx
	movq	-112(%rbp), %rax
	movb	%dl, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L8:
	movl	-80(%rbp), %eax
	movl	%eax, -64(%rbp)
	addl	$1, -80(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, %ecx
	movq	-112(%rbp), %rdx
	movl	-64(%rbp), %eax
	cltq
	movb	%cl, 4(%rdx,%rax)
	movq	$1, -8(%rbp)
	jmp	.L45
.L20:
	cmpl	$98, -68(%rbp)
	jg	.L74
	movq	$19, -8(%rbp)
	jmp	.L45
.L74:
	movq	$54, -8(%rbp)
	jmp	.L45
.L19:
	movq	-112(%rbp), %rax
	movl	$11, (%rax)
	movq	-112(%rbp), %rax
	movb	$0, 4(%rax)
	movq	$8, -8(%rbp)
	jmp	.L45
.L6:
	movq	-112(%rbp), %rax
	movl	$4, (%rax)
	movq	-112(%rbp), %rax
	movb	$43, 4(%rax)
	movq	$29, -8(%rbp)
	jmp	.L45
.L23:
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L45
.L28:
	movq	-112(%rbp), %rax
	movb	$0, 5(%rax)
	movq	$4, -8(%rbp)
	jmp	.L45
.L35:
	call	__ctype_b_loc@PLT
	movq	%rax, -32(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L45
.L77:
	nop
.L45:
	jmp	.L76
.L78:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	getNextToken, .-getNextToken
	.section	.rodata
.LC2:
	.string	"Token: %d, Text: %s\n"
.LC3:
	.string	"Failed to open file"
.LC4:
	.string	"r"
.LC5:
	.string	"input.txt"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_i3df_envp(%rip)
	nop
.L80:
	movq	$0, _TIG_IZ_i3df_argv(%rip)
	nop
.L81:
	movl	$0, _TIG_IZ_i3df_argc(%rip)
	nop
	nop
.L82:
.L83:
#APP
# 90 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-i3df--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_i3df_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_i3df_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_i3df_envp(%rip)
	nop
	movq	$11, -128(%rbp)
.L101:
	cmpq	$11, -128(%rbp)
	ja	.L104
	movq	-128(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L86(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L86(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L86:
	.long	.L94-.L86
	.long	.L93-.L86
	.long	.L92-.L86
	.long	.L91-.L86
	.long	.L90-.L86
	.long	.L104-.L86
	.long	.L104-.L86
	.long	.L89-.L86
	.long	.L88-.L86
	.long	.L87-.L86
	.long	.L104-.L86
	.long	.L85-.L86
	.text
.L90:
	movl	-112(%rbp), %eax
	cmpl	$11, %eax
	je	.L95
	movq	$1, -128(%rbp)
	jmp	.L97
.L95:
	movq	$8, -128(%rbp)
	jmp	.L97
.L88:
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$0, -128(%rbp)
	jmp	.L97
.L93:
	leaq	-112(%rbp), %rdx
	movq	-136(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	getNextToken
	movl	-112(%rbp), %eax
	leaq	-112(%rbp), %rdx
	addq	$4, %rdx
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -128(%rbp)
	jmp	.L97
.L91:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$9, -128(%rbp)
	jmp	.L97
.L85:
	movq	$2, -128(%rbp)
	jmp	.L97
.L87:
	movl	$1, %eax
	jmp	.L102
.L94:
	movl	$0, %eax
	jmp	.L102
.L89:
	cmpq	$0, -136(%rbp)
	jne	.L99
	movq	$3, -128(%rbp)
	jmp	.L97
.L99:
	movq	$1, -128(%rbp)
	jmp	.L97
.L92:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, -136(%rbp)
	movq	$7, -128(%rbp)
	jmp	.L97
.L104:
	nop
.L97:
	jmp	.L101
.L102:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L103
	call	__stack_chk_fail@PLT
.L103:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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
